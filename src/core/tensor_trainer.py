"""Listwise training of the query-adaptive channel weights (Blueprint 5.1, v2).

Fairness contract (unchanged):
- trains ONLY on the dev question pool (generate_paper_protocol_questions
  with pool="dev"), which is article/entity-hash-disjoint from the
  evaluation (test) pool;
- no LLM judge in the training loop: the supervision signal is the binary
  gold-chunk relevance of each retrieved candidate, so the judge cannot
  bias what is learned;
- the learned weights are saved to models/dynamic_tensor.pt and loaded only
  when HCAD_TENSOR_MODE=learned; the static heuristic remains the default.

Why this replaced the REINFORCE loop: the previous trainer was
mathematically incapable of learning. Its score function summed
simplex-normalized differences, gl = sum((lam_s - mu_lam)/sigma^2), which is
identically zero because both vectors lie on the unit simplex - and the
sampled action was never detached into a log-density surrogate. The model
backpropagated floating-point noise (Critical Research Review, gradient
collapse proof).

What trains now: a low-rank ChannelWeightAdapter (~78 parameters) that
predicts the 3 component weights a(Q) = (alpha, beta, gamma) from the
25-dimensional normalized query features, with lambda held isotropic
(uniform 1/d). Per-channel similarities S(Q, C_i) do not depend on a, so
each dev question yields a fixed (N, 3) matrix and the listwise InfoNCE
loss over Phi = S @ a has exact, low-variance gradients. The 768-dim
lambda head of the old hypernetwork is deliberately dropped: its block
weights largely cancel in the weighted-cosine denominator and a ~200K-
parameter hypernetwork overfits 60-question dev pools (Critical Review,
rank-8 recommendation).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.config import (
    SCORE_WEIGHT_ENTITY,
    SCORE_WEIGHT_MODIFIER,
    SCORE_WEIGHT_RELATION,
    presence_masking_enabled,
)

logger = logging.getLogger(__name__)


def _mean_reward(rewards: list[float], epoch: int) -> float:
    if not rewards:
        raise RuntimeError(
            f"Tensor training epoch {epoch} produced no rewards; "
            "all dev questions were skipped during retrieval."
        )
    return sum(rewards) / len(rewards)


def train_tensor(
    retriever,
    dev_questions: list[dict],
    epochs: int = 3,
    lr: float = 5e-3,
    k: int = 5,
    tau: float = 0.15,
    save_path: Path | None = None,
) -> dict:
    """Train the query-adaptive channel weights on dev-pool questions.

    Args:
        retriever: a constructed HCADRetriever. The warm-up pass runs under
            an explicit isotropic-lambda / static-weights override so the
            retriever captures feats and the per-channel similarity matrix
            (its learned branch records both - see HCADRetriever).
        dev_questions: paper-protocol questions with pool="dev"
            (each carries "question" and "reference").
        epochs: passes over the dev set.
        lr: Adam learning rate for the adapter head.
        k: NDCG cut for the per-epoch telemetry reward.
        tau: InfoNCE temperature over Phi scores.
        save_path: checkpoint destination (default models/dynamic_tensor.pt).

    Returns:
        dict with per-epoch telemetry (mean NDCG reward, mean gradient
        norm) and the checkpoint path.
    """
    from src.core.dynamic_tensor import ChannelWeightAdapter, save_channel_adapter
    from src.evaluation.metrics import ndcg_at_k
    from src.evaluation.run_base_paper_benchmark import _ranked_relevance

    if not dev_questions:
        raise ValueError("dev_questions is empty - generate the dev pool first "
                         "(generate_paper_protocol_questions(pool='dev')).")
    pools = {q.get("pool", "unknown") for q in dev_questions}
    if pools - {"dev"}:
        raise ValueError(
            f"Refusing to train on non-dev question pools: {pools}. "
            "Training on test-pool questions would leak the evaluation set."
        )

    embed_dim = 768
    uniform_lam = torch.full((embed_dim,), 1.0 / embed_dim, dtype=torch.float32)
    static_a = torch.tensor(
        [SCORE_WEIGHT_ENTITY, SCORE_WEIGHT_RELATION, SCORE_WEIGHT_MODIFIER],
        dtype=torch.float32)

    # Feature-parity guard: the adapter learns query features derived from
    # NER output. Training on the regex fallback while inference (session-3
    # ablation) runs scispacy would learn a mismatched feature distribution
    # - fail loudly instead of silently training on degraded features.
    from src.core.ner_extractor import get_biomedical_ner
    ner = get_biomedical_ner()
    if getattr(ner, "degraded", False):
        raise RuntimeError(
            "SciSpacy NER failed to load (BiomedicalNER in degraded regex "
            "mode). Refusing to train: the channel-weight adapter would "
            "learn a feature distribution that differs from inference. "
            "Re-run training; the loader now retries transient failures."
        )

    adapter = ChannelWeightAdapter(feat_dim=25)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    retriever._tensor = None
    retriever._tensor_override = None

    history: list[dict] = []
    for epoch in range(epochs):
        rewards: list[float] = []
        grad_norms: list[float] = []
        for item in dev_questions:
            question = item["question"]
            reference = item["reference"]

            # 1) warm-up retrieval under uniform lambda + static channel
            #    weights: the retriever's learned branch records
            #    (Q, feats) and the per-channel similarity matrix for
            #    this (query, candidate pool) pair.
            retriever._tensor_override = (uniform_lam, static_a)
            try:
                contexts = retriever.retrieve(question)
            except Exception as exc:
                logger.warning("Warm-up retrieval failed for a dev question "
                               "(%s); skipping it.", exc)
                continue
            finally:
                retriever._tensor_override = None
            inputs = getattr(retriever, "_last_tensor_inputs", None)
            masked_sims = getattr(retriever, "_last_masked_sims", None)
            if presence_masking_enabled() and masked_sims is not None:
                sims = masked_sims
                q_presence = getattr(retriever, "_last_query_presence", None)
            else:
                sims = getattr(retriever, "_last_channel_sims", None)
                q_presence = None
            pool = getattr(retriever, "_last_candidate_pool", None)
            if inputs is None or sims is None or not pool:
                continue
            # The sims matrix is computed over the PRE-GATE candidate pool;
            # the listwise loss therefore runs over that same pool (gold
            # may be present but gated out - the loss must be able to teach
            # the channel weights to rank it back in). Hard alignment guard:
            # rows must match the pool exactly or the pair is unusable.
            if len(pool) != len(np.asarray(sims)):
                logger.warning(
                    "Channel sims (%d rows) misaligned with candidate pool "
                    "(%d items); skipping this question.",
                    len(np.asarray(sims)), len(pool))
                continue
            _Q_t, feats = inputs

            # 2) binary gold relevance per candidate (no LLM judge).
            relevance = _ranked_relevance(reference, pool)
            positives = [i for i, r in enumerate(relevance) if r]
            if not positives:
                # The gold chunk is not in this candidate pool: the pair
                # carries no listwise signal for the channel weights.
                continue

            S = torch.tensor(np.asarray(sims, dtype=np.float32))   # (N, 3)
            y = torch.tensor([1.0 if r else 0.0 for r in relevance],
                             dtype=torch.float32)                    # (N,)
            # Train/serve parity with the hybrid score: inference ranks by
            # Phi = w_d * dense + (1 - w_d) * Phi_struct, so the listwise
            # loss must see the same fixed dense term (no gradient flows
            # through it; it shifts which structural weights win).
            dense_sims = getattr(retriever, "_last_dense_sims", None)
            w_d = float(getattr(retriever, "_last_dense_weight", 0.0) or 0.0)
            if dense_sims is not None and len(np.asarray(dense_sims)) == len(pool) and w_d > 0.0:
                dense_t = torch.tensor(np.asarray(dense_sims, dtype=np.float32))
            else:
                dense_t, w_d = None, 0.0

            # 3) exact differentiable listwise loss: push the adapter's
            #    Phi = S @ a(Q) above the pool's non-gold candidates.
            #    When presence masking is enabled, match inference weighting:
            #    omega = (a * p_Q) / sum(a * p_Q)
            a = adapter(feats)                       # (3,)
            if q_presence is not None:
                p_Q = torch.tensor(
                    np.asarray(q_presence, dtype=np.float32).ravel(),
                    device=a.device,
                    dtype=a.dtype,
                )
                active = a * p_Q
                denom = active.sum()
                if float(denom.detach()) <= 0.0:
                    omega = torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype)
                else:
                    omega = active / denom
                phi = S @ omega
            else:
                phi = S @ a                               # (N,)
            if dense_t is not None:
                phi = w_d * dense_t + (1.0 - w_d) * phi
            logp = torch.log_softmax(phi / tau, dim=-1)
            loss = -logp[positives].mean()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), max_norm=1.0)
            grad_norms.append(float(grad_norm))
            optimizer.step()

            # 4) telemetry: NDCG@k of the pool re-ranked by the CURRENT
            #    channel weights (comparable to the old trainer's reward).
            order = torch.argsort(phi.detach(), descending=True)
            ranked_rel = [float(y[i]) for i in order]
            rewards.append(float(ndcg_at_k(ranked_rel, k)))

        mean_reward = _mean_reward(rewards, epoch + 1)
        mean_grad = sum(grad_norms) / max(len(grad_norms), 1)
        history.append({"epoch": epoch + 1, "mean_ndcg_reward": mean_reward,
                        "n": len(rewards), "mean_grad_norm": round(mean_grad, 6)})
        logger.info("Tensor training epoch %d/%d: mean NDCG@%d reward %.4f "
                    "(%d questions, mean grad-norm %.4f)",
                    epoch + 1, epochs, k, mean_reward, len(rewards), mean_grad)

    retriever._tensor = None  # trainer never leaks the active tensor
    path = save_channel_adapter(adapter, save_path)
    logger.info("Trained channel-weight adapter saved to %s", path)
    return {"epochs": history, "checkpoint": str(path),
            "n_trained": len(rewards)}


def load_tensor_for_ablation(retriever) -> bool:
    """Activate the learned tensor for an ablation benchmark run.

    Used by the benchmark harness / notebook when HCAD_TENSOR_MODE=learned;
    returns True when the learned branch is active.
    """
    return retriever._get_tensor() is not None
