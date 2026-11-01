"""
Benchmark Runner replicating Base Paper  evaluation protocol.

Compares:
1. Baseline LLM (No Retrieval)
2. Vanilla Vector-RAG (Standard Dense Top-K)
3. HCAD-RAG (Full Proposed Architecture)

Outputs:
- 5-Dimension Scoring-Based Assessment (0-100)
- 8-Criteria Selection-Based Win-Rate (%)
- Retrieval Latency Benchmark (seconds)
- Robustness metrics (src/evaluation/metrics.py):
  - NDCG@K / Recall@K over paper-protocol gold relevance labels
  - McNemar exact significance test on paired correctness
  - Citation precision of [Evidence i] tags against retrieved items
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVIDENCE_WORD_BUDGET, RANDOM_SEED, RESULTS_DIR
from src.core.prompt_builder import context_document
from src.evaluation.base_paper_evaluator import BasePaperEvaluator
from src.evaluation.metrics import citation_precision, mcnemar_test, ndcg_at_k, recall_at_k
from src.wandb_tracking import log_base_paper_results, tracked_run, upload_checkpoint_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_clinical_benchmark_samples(num_samples: int = 5) -> List[Dict[str, str]]:
    """Load gold-standard clinical benchmark cases matching the base paper  protocol.

    Attempts to load from dataset files first, falls back to hardcoded samples if not available.
    """
    import json
    from pathlib import Path

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    # Try to load from processed dataset files
    try:
        # Look for MedQA or other clinical datasets
        medqa_path = Path(__file__).parent.parent.parent / "data" / "processed" / "medqa_samples.json"
        if medqa_path.exists():
            with open(medqa_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            if isinstance(samples, list) and len(samples) >= num_samples:
                for sample in samples:
                    if isinstance(sample, dict):
                        sample.setdefault("source", "medqa_processed")
                logger.info(f"Loaded {len(samples)} clinical samples from {medqa_path}")
                return samples[:num_samples]
    except Exception as e:
        logger.debug(f"Could not load clinical samples from file: {e}")

    # Fallback to hardcoded gold-standard clinical cases
    clinical_cases = [
        {
            "question": "How should long-term treatment and motor symptom recovery of patients with Parkinson's disease and tremor be managed?",
            "reference": "Parkinson's disease (PD) motor management is rooted in restoring striatal dopaminergic signaling. First-line pharmacotherapy uses carbidopa/levodopa as the most effective symptomatic agent, supplemented by dopamine receptor agonists (pramipexole, ropinirole) and MAO-B inhibitors (rasagiline) or COMT inhibitors (entacapone) to smooth out motor fluctuations and wearing-off phenomena. For medication-refractory resting tremors and disabling motor fluctuations, surgical intervention with Deep Brain Stimulation (DBS) targeting the subthalamic nucleus (STN) or globus pallidus internus (GPi) yields significant motor improvement and reduces medication requirements. Comprehensive care also incorporates physical therapy, fall risk mitigation, and neurorehabilitation.",
            "ground_truth": "Levodopa pharmacotherapy combined with deep brain stimulation (DBS) for refractory motor tremors."
        },
        {
            "question": "What are the emergency management and risk stratification guidelines for acute ST-elevation myocardial infarction (STEMI)?",
            "reference": "Acute ST-elevation myocardial infarction (STEMI) represents complete transmural myocardial ischemia typically caused by acute coronary atherothrombotic occlusion. Emergency management mandates immediate reperfusion therapy: primary percutaneous coronary intervention (PCI) is preferred if door-to-balloon time is under 90 minutes (or under 120 minutes if transferred); intravenous fibrinolysis is indicated within 30 minutes if PCI is unavailable within 120 minutes. Concomitant pharmacotherapy includes immediate dual antiplatelet therapy (aspirin plus a potent P2Y12 inhibitor such as ticagrelor or prasugrel), unfractionated heparin or bivalirudin anticoagulation, sublingual nitroglycerin for ischemic pain, and high-intensity statin therapy. Risk stratification incorporates TIMI or GRACE risk scores and assessment for cardiogenic shock or mechanical complications.",
            "ground_truth": "Immediate primary PCI within 90 minutes or fibrinolysis within 30 minutes, combined with dual antiplatelet therapy, anticoagulation, and statins."
        },
        {
            "question": "What is the stepwise diagnostic and therapeutic algorithm for acute severe exacerbation of bronchial asthma?",
            "reference": "Acute severe asthma exacerbation is marked by acute bronchospasm, airway inflammation, and ventilation-perfusion mismatch. Emergency therapeutic protocol begins with high-concentration supplemental oxygen (titrated to SpO2 93-95%), repetitive or continuous nebulized short-acting beta-2 agonists (albuterol/salbutamol) combined with inhaled anticholinergics (ipratropium bromide), and early administration of systemic corticosteroids (oral prednisone or IV methylprednisolone) to reduce airway edema. For refractory severe bronchospasm, intravenous magnesium sulfate is administered. Clinical monitoring includes serial peak expiratory flow (PEF) measurements, arterial blood gas analysis to detect hypercapnia and impending respiratory muscle fatigue, and preparation for non-invasive positive pressure ventilation or endotracheal intubation if respiratory arrest is imminent.",
            "ground_truth": "Continuous inhaled beta-agonists plus ipratropium, early systemic corticosteroids, supplemental oxygen, and IV magnesium sulfate for refractory bronchospasm."
        },
        {
            "question": "How should cognitive and behavioral symptoms in moderate-to-severe Alzheimer's dementia be evaluated and managed?",
            "reference": "Alzheimer's disease management encompasses cognitive stabilization, neuropsychiatric symptom control, and caregiver support. Pharmacological therapy for moderate-to-severe dementia combines a cholinesterase inhibitor (donepezil, rivastigmine, or galantamine) with the NMDA receptor antagonist memantine, which provides neuroprotection against excitotoxicity. Behavioral and psychological symptoms of dementia (BPSD) including agitation, psychosis, and sleep disruption require initial non-pharmacological behavioral modifications and environmental optimization. Low-dose atypical antipsychotics are reserved strictly for severe distressing agitation or psychosis unresponsive to non-pharmacological measures, with close cardiac and metabolic monitoring due to black box mortality warnings.",
            "ground_truth": "Combination of cholinesterase inhibitors and memantine with non-pharmacological behavioral management for neuropsychiatric symptoms."
        },
        {
            "question": "What are the diagnostic criteria and management strategies for type 2 diabetes mellitus?",
            "reference": "Type 2 diabetes mellitus is characterized by insulin resistance and relative insulin deficiency. Diagnostic criteria include HbA1c ≥6.5%, fasting plasma glucose ≥126 mg/dL, 2-hour plasma glucose ≥200 mg/dL during OGTT, or random plasma glucose ≥200 mg/dL with classic symptoms. Management focuses on lifestyle modification with medical nutrition therapy, physical activity, and pharmacotherapy. First-line therapy is metformin, with addition of other agents (SGLT2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, sulfonylureas, or insulin) based on HbA1c targets, patient comorbidities, and cost considerations. Comprehensive management includes cardiovascular risk reduction, blood pressure control, lipid management, and regular monitoring for complications.",
            "ground_truth": "Metformin as first-line therapy with comprehensive cardiovascular risk reduction and regular complication monitoring."
        }
    ]
    for sample in clinical_cases:
        sample.setdefault("source", "curated")

    # the MedQA fallback was removed - the base paper never
    # uses MCQ benchmarks, and mixing question formats made aggregate scores
    # heterogeneous. Oversize requests are truncated to the curated pool.
    return clinical_cases[:num_samples]

def _merge_selection_results(ab_result: Dict, ba_result: Dict, target_label: str) -> Dict:
    """Merge AB and BA pairwise evaluations to counterbalance position bias."""
    merged_criteria = {}
    ab_criteria = ab_result.get("criteria_winners", {})
    ba_criteria = ba_result.get("criteria_winners", {})

    for crit in BasePaperEvaluator.CRITERIA_LIST:
        ab_winner = ab_criteria.get(crit, "Tie")
        ba_winner = ba_criteria.get(crit, "Tie")
        # If both orders agree on the same winner, keep it. Otherwise, Tie.
        if ab_winner == ba_winner:
            merged_criteria[crit] = ab_winner
        else:
            merged_criteria[crit] = "Tie"

    ab_overall = ab_result.get("overall_winner", "Tie")
    ba_overall = ba_result.get("overall_winner", "Tie")
    if ab_overall == ba_overall:
        merged_overall = ab_overall
    else:
        merged_overall = "Tie"

    return {
        "criteria_winners": merged_criteria,
        "overall_winner": merged_overall,
        "summary": f"AB: {ab_result.get('summary', '')} | BA: {ba_result.get('summary', '')}",
    }

def _validate_generated_response(system_name: str, response: str) -> str:
    """Stop the benchmark instead of scoring an error message as an answer."""
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"{system_name} returned an empty or non-text response")
    cleaned = response.strip()
    if cleaned.startswith("[Error:"):
        raise RuntimeError(f"{system_name} generation failed: {cleaned}")
    return cleaned

_ARM_CITATION_TAG = re.compile(r"\[(?:Evidence|Context)(\s*\d[\d,\s\-–]*)\]")

def _blind_citation_tags(response: str) -> str:
    """Rewrite arm-specific citation tags to the judge's neutral [Doc i] tags.

    Vanilla answers cite [Context i] and HCAD answers cite [Evidence i]; the
    judge sees both evidence blocks tagged [Doc i]. Left as-is, the tag word
    reveals which system wrote an answer (unblinding the pairwise judge) and
    makes citations harder to match to evidence. Indices are unchanged, so
    [Doc i] still points at the same item. Applied only to judge inputs;
    citation precision is computed on the raw response.
    """
    return _ARM_CITATION_TAG.sub(r"[Doc\1]", response)

def _format_retrieved_evidence(contexts: List[Dict[str, Any]], tag: str) -> str:
    """Expose answer-specific retrieval evidence to the judge with a word cap."""
    if not contexts:
        return "No retrieved evidence was supplied for this answer."

    lines = []
    remaining_words = EVIDENCE_WORD_BUDGET
    for index, context in enumerate(contexts, 1):
        metadata = context.get("metadata") or {}
        document = context.get("document") or context.get("text") or ""
        if not document:
            head = metadata.get("head_name", "")
            relation = metadata.get("display_relation") or metadata.get("relation", "")
            tail = metadata.get("tail_name", "")
            document = f"{head} {relation} {tail}".strip()
        words = str(document).split()
        if not words or remaining_words <= 0:
            break
        # clip at the last sentence boundary within the
        # budget - mid-sentence clipping can sever negations/qualifiers.
        budget_text = " ".join(words[:remaining_words])
        cut = max(budget_text.rfind(". "), budget_text.rfind("! "), budget_text.rfind("? "))
        selected = budget_text if cut == -1 else budget_text[: cut + 1]
        lines.append(f"[{tag} {index}] {selected.strip()}")
        used = len(selected.split())
        remaining_words -= used
        if used < len(words):
            break
    return "\n".join(lines) or "Retrieved contexts contained no readable evidence."

# Both systems retrieve top-5; all retrieval metrics use the same cut.
RETRIEVAL_K = 5
# Paired correctness threshold for McNemar: overall_score >= 70 of 100.
CORRECTNESS_THRESHOLD = 70.0

_STOPWORDS = frozenset(
    "the a an and or of to in for with on by is are was were be been this that "
    "these those it its as at from may can which who whom what when where how "
    "not no than then also such into during before after above below over under "
    "between within without about against each other more most less least any "
    "all some both few many much very own same so if while because through "
    "their there here they them we you he she his her our your my me us him".split()
)

_context_document = context_document

def _content_tokens(text: str) -> set:
    """Lowercased content tokens; stop- and short tokens dropped."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }

def _ranked_relevance(
    reference: str,
    contexts: List[Dict[str, Any]],
    overlap_threshold: float = 0.3,
) -> List[int]:
    """Binary relevance of retrieved contexts against the gold source chunk.

    The paper protocol generates each question FROM a source chunk, so that
    chunk is the free gold relevance label. A retrieved context is relevant
    when its content-token Jaccard overlap with the reference reaches the
    threshold (1.0 for the chunk itself; paraphrases stay well above it
    while short fragments sharing a word or two do not).
    """
    ref_tokens = _content_tokens(reference)
    relevance: List[int] = []
    if not ref_tokens:
        return [0] * len(contexts)
    for context in contexts:
        doc_tokens = _content_tokens(_context_document(context))
        union = len(ref_tokens | doc_tokens)
        overlap = len(ref_tokens & doc_tokens) / union if union else 0.0
        relevance.append(1 if overlap >= overlap_threshold else 0)
    return relevance

def _records_from_checkpoint(
    checkpoint_path: Path, sample_questions: set, question_pool: Optional[str] = None,
    question_set_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the newest per-question record for every question in this run.

    Aggregating from the checkpoint (not session memory) keeps resumed runs
    correct: questions evaluated before the interruption keep their records,
    and stale questions from older runs with different sample sets are
    filtered out.
    """
    by_question: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (record.get("question") in sample_questions
                        and (question_pool is None
                             or record.get("question_pool") == question_pool)):
                    if (question_set_id is None
                            or record.get("question_set_id") == question_set_id):
                        by_question[record["question"]] = record  # later lines win

    scoring = {"Baseline LLM": [], "Vanilla RAG": [], "HCAD-RAG": []}
    selection = {"HCAD_vs_Vanilla": [], "HCAD_vs_Baseline": [],
                 "Vanilla_vs_Baseline": []}
    latency = {"Vanilla RAG": [], "HCAD-RAG": []}
    retrieval = {"Vanilla RAG": [], "HCAD-RAG": []}
    citations = {"Vanilla RAG": [], "HCAD-RAG": []}
    scoring_by_tier: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    scoring_by_source: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    selection_by_source: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for record in by_question.values():
        scores = record.get("scores") or {}
        for system, score in scores.items():
            if system in scoring:
                scoring[system].append(score)
                tier_records = scoring_by_tier.setdefault(
                    int(record.get("difficulty_tier", 1) or 1),
                    {sys: [] for sys in scoring},
                )
                tier_records[system].append(score)
                source_records = scoring_by_source.setdefault(
                    record.get("source", "curated"),
                    {sys: [] for sys in scoring},
                )
                source_records[system].append(score)
        for comparison, sel in (record.get("selection") or {}).items():
            if comparison in selection:
                selection[comparison].append(sel)
                sel_source = selection_by_source.setdefault(
                    record.get("source", "curated"),
                    {comparison: [] for comparison in selection},
                )
                sel_source[comparison].append(sel)
        for system, lat in (record.get("latency") or {}).items():
            if system in latency:
                latency[system].append(lat)
        for system, ret in (record.get("retrieval") or {}).items():
            if system in retrieval and ret:
                retrieval[system].append(ret)
        for system, cit in (record.get("citations") or {}).items():
            if system in citations:
                citations[system].append(cit)

    return {
        "scoring": scoring,
        "selection": selection,
        "latency": latency,
        "retrieval": retrieval,
        "citations": citations,
        "scoring_by_tier": scoring_by_tier,
        "scoring_by_source": scoring_by_source,
        "selection_by_source": selection_by_source,
        "per_question": list(by_question.values()),
    }

QUESTION_GEN_PROMPT = (
    "You are given a passage from a clinical knowledge base. Generate one "
    "examination-style, open-ended question that this passage fully answers.\n"
    "{tier}\n"
    "Return ONLY the question text.\n\nPassage:\n{chunk}"
)
QUESTION_GEN_PROMPT_ASSOC = (
    "You are given a single factual statement from a biomedical knowledge graph "
    "describing one association (for example, between a drug and a protein, a "
    "disease and a phenotype, or two drugs). Generate one examination-style, "
    "open-ended question that this statement fully answers.\n"
    "Rules: ask only about what the statement itself supports; do NOT ask about "
    "mechanisms, processes, or clinical management that the statement does not "
    "describe. Any follow-up aspect must remain answerable from the statement.\n"
    "{tier}\n"
    "Return ONLY the question text.\n\nStatement:\n{chunk}"
)
TIERS = {
    1: "Ask a single focused question.",
    2: "Ask a two-stage question: first one aspect, then 'and specifically' a second, dependent aspect.",
    3: "Ask a three-stage nested question chaining three dependent aspects with connecting phrases.",
}

# Stratified sampling: question sources are drawn across these PrimeKG relation
# types plus the PMC clinical-text corpus, instead of the first N documents of
# the vector store (which are all contiguous PrimeKG drug_protein triples -
# the root cause of the degenerate 8-fact pilot question set).
# Relation vocabulary verified against the built store's sqlite
# (data/vectordb/chroma.sqlite3): contraindication, disease_disease,
# disease_phenotype_negative, disease_phenotype_positive, disease_protein,
# drug_effect, drug_protein, indication, off-label use. Strata absent from
# the store would simply be skipped, but keeping the list exact avoids
# wasted fetch attempts and guarantees every stratum contributes.
TRIPLE_RELATION_STRATA = [
    "drug_protein", "contraindication", "indication", "drug_effect",
    "disease_protein", "disease_phenotype_positive",
    "disease_phenotype_negative", "disease_disease", "off-label use",
]

def _question_pool_of(doc_id: str, pool: str) -> bool:
    """Deterministic dev/test pool assignment by article/entity key hash.

    The key is the SAME one used for per-chunk dedup: the parent article
    (doc_id) for PMC prose and the head entity (head_name) for PrimeKG
    triples. This keeps the pools disjoint at the article/entity level, so
    no single clinical study or drug contributes questions to both the
    pool used to train/tune HCAD components and the held-out evaluation
    pool (no data leakage, Critical Review V3).
    """
    import hashlib
    h = int(hashlib.sha1(str(doc_id).encode("utf-8")).hexdigest()[:8], 16)
    want = 0 if pool == "test" else 1
    return h % 2 == want

def _stratum_size(collection, where: dict, upper_bound: int) -> Optional[int]:
    """Find the filtered row count using offset probes, or None if unsupported."""
    low, high = 0, upper_bound
    while low < high:
        mid = (low + high) // 2
        try:
            got = collection.get(where=where, limit=1, offset=mid,
                                 include=["documents", "metadatas"])
        except TypeError:
            return None
        if (got or {}).get("documents"):
            low = mid + 1
        else:
            high = mid
    return low

def _paper_question_cache_path(num_samples: int, pool: str) -> Path:
    if pool not in ("test", "dev"):
        raise ValueError(f"pool must be 'test' or 'dev', got {pool!r}")
    return RESULTS_DIR / f"paper_questions_n{num_samples}_seed{RANDOM_SEED}_{pool}.json"

def _question_set_id(samples: list[dict]) -> str:
    import hashlib

    contract = [(sample["question"], sample.get("source_chunk_id"), sample.get("pool"))
                for sample in samples]
    payload = json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _completed_questions_from_checkpoint(
    checkpoint_path: Path, pool: str, question_set_id: str
) -> set[str]:
    completed: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A killed session can leave a truncated final line; the
                    # aggregation path already skips those - resume must not
                    # crash on the exact failures it exists to recover from.
                    continue
                if (record.get("question_pool") == pool
                        and record.get("question_set_id") == question_set_id):
                    completed.add(record["question"])
    return completed

def _grounded_mode_counts(
    checkpoint_path: Path, pool: str, question_set_id: str
) -> Dict[str, int]:
    """P2d provenance: grounded vs parametric HCAD answers for this sample set.

    Read from the checkpoint so resumed runs aggregate the full set, not
    just the questions evaluated in the current session.
    """
    counts = {"grounded": 0, "parametric": 0}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (record.get("question_pool") == pool
                        and record.get("question_set_id") == question_set_id):
                    mode = record.get("grounded_mode")
                    if mode in counts:
                        counts[mode] += 1
    return counts

def _fetch_stratified_chunks(collection, num_docs: int, pool: str, rng) -> list[dict]:
    """Fetch candidate source chunks stratified by corpus segment.

    PMC clinical-text chunks and each PrimeKG relation type form separate
    strata. Within each stratum MULTIPLE random windows are drawn,
    round-robin across strata, until enough distinct pool-qualified chunks
    are collected: PrimeKG's kg.csv is ordered by head drug, so a single
    contiguous window is almost one entity and the old one-window-per-
    stratum draw produced only ~50 usable chunks from the 817k store.
    Each stratum's offset span shrinks adaptively when a window lands
    past the end of that stratum. Chunks are deduplicated per entity
    (head_name) / article (doc_id) so a single drug cannot dominate, and
    filtered to the requested dev/test pool.
    """
    strata = [{"where": {"source": "pmc"}, "kind": "pmc",
               "span": None, "no_offset": False}]
    strata += [{"where": {"relation": rel}, "kind": "primekg",
                "span": None, "no_offset": False}
               for rel in TRIPLE_RELATION_STRATA]
    total = collection.count()
    for s in strata:
        s["span"] = _stratum_size(collection, s["where"], total)
        if s["span"] is None:
            s["no_offset"] = True
    target = num_docs * 2
    per = 64
    out: list[dict] = []
    seen: set[str] = set()
    for _round in range(24):
        if len(out) >= target:
            break
        for s in strata:
            if len(out) >= target:
                break
            if s["no_offset"]:
                try:
                    got = collection.get(where=s["where"], limit=per,
                                         include=["documents", "metadatas"])
                except TypeError:
                    got = None
            else:
                span = s["span"]
                if not span:
                    continue
                offset = rng.randrange(0, max(1, span - per + 1))
                try:
                    got = collection.get(where=s["where"], limit=per,
                                         offset=offset,
                                         include=["documents", "metadatas"])
                except TypeError:
                    # chroma build without offset support: head window only
                    s["no_offset"] = True
                    try:
                        got = collection.get(where=s["where"], limit=per,
                                             include=["documents", "metadatas"])
                    except TypeError:
                        got = None
                else:
                    if not (got or {}).get("documents") and offset > 0:
                        # window landed past this stratum's end - shrink span
                        s["span"] = max(per, offset)
                        continue
            docs = (got or {}).get("documents") or []
            metas = (got or {}).get("metadatas") or [{}] * len(docs)
            ids = (got or {}).get("ids") or [str(i) for i in range(len(docs))]
            for d, m, i in zip(docs, metas, ids):
                if not d or len(d) < 30:
                    continue
                if s["kind"] == "pmc" and len(d) < 200:
                    continue
                if s["kind"] == "primekg" and len(d) > 300:
                    continue
                key = str(m.get("doc_id") or m.get("head_name") or i)
                if key in seen:
                    continue
                # Pool assignment must run on the SAME key as dedup (article
                # doc_id for PMC, head entity for PrimeKG): hashing raw chunk
                # ids let chunks of one article / one drug straddle the dev
                # and test pools (Critical Review V3 leakage vector).
                if not _question_pool_of(key, pool):
                    continue
                seen.add(key)
                out.append({"doc": d, "id": str(i), "kind": s["kind"],
                            "relation": str(m.get("relation") or ""),
                            # the article/entity key the pool was assigned by
                            # (provenance for leakage audits and for tests)
                            "pool_key": key})
    rng.shuffle(out)
    return out

def generate_paper_protocol_questions(num_samples: int, generator=None,
                                      pool: str = "test") -> list[dict]:
    """Base-paper protocol (Sec. 3.5): questions are GENERATED from source
    chunks by an LLM, and the SOURCE CHUNK is the scoring reference.

    Fairness properties:
    - chunks are drawn stratified across PMC text and PrimeKG relation types
      (not the head of the store), deduplicated per entity/article;
    - each question comes from a DISTINCT chunk (tiers get disjoint slices -
      previously every tier rephrased the same 8 facts);
    - chunks are assigned to the dev or test pool by id hash, so training/
      tuning on the dev pool can never leak into the evaluation pool;
    - questions from short KG-triple statements use an association-style
      prompt that forbids asking mechanisms the statement cannot answer.

    The generated set is CACHED to disk (keyed by sample count, seed, pool).
    Generation samples at temperature 0.3, so without the cache a resumed
    run would produce different questions and silently discard every
    checkpointed answer. With it, interrupted runs resume exactly.
    """
    from src.core.llm_client import UnifiedLLMClient
    from src.data.build_vector_store import load_vector_store
    import random as _random

    if pool not in ("test", "dev"):
        raise ValueError(f"pool must be 'test' or 'dev', got {pool!r}")
    cache_path = _paper_question_cache_path(num_samples, pool)
    legacy_path = RESULTS_DIR / f"paper_questions_n{num_samples}_seed{RANDOM_SEED}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (len(cached) == num_samples and all(
                    "question" in s and "reference" in s
                    and s.get("pool") == pool
                    and s.get("source") == "paper_protocol"
                    and s.get("source_chunk_id")
                    for s in cached)):
                logger.info("Paper-protocol questions loaded from cache: %d (pool=%s).",
                            len(cached), pool)
                return cached
        except Exception as exc:
            logger.warning("Question cache unreadable (%s); regenerating.", exc)
    if legacy_path.exists():
        logger.warning("Ignoring unscoped legacy question cache %s; regenerating pool=%s.",
                       legacy_path.name, pool)

    if generator is None:
        generator = UnifiedLLMClient(temperature=0.3)
    collection = load_vector_store()
    total = collection.count()
    if total == 0:
        raise RuntimeError("Vector store is empty - run `python src/main.py prepare` first.")
    rng = _random.Random(RANDOM_SEED)
    chunks = _fetch_stratified_chunks(collection, num_samples * 2, pool, rng)
    if len(chunks) < num_samples:
        raise RuntimeError(
            f"Only {len(chunks)} distinct pool-{pool} source chunks available "
            f"for {num_samples} questions; regenerate with a different pool size."
        )

    per_tier = max(1, -(-num_samples // 3))
    samples, tier_sizes = [], {1: 0, 2: 0, 3: 0}
    cursor = 0
    for tier in (1, 2, 3):
        while tier_sizes[tier] < per_tier and len(samples) < num_samples:
            if cursor >= len(chunks):
                raise RuntimeError("Ran out of distinct source chunks during generation.")
            chunk = chunks[cursor]
            cursor += 1
            prompt = (QUESTION_GEN_PROMPT if chunk["kind"] == "pmc"
                     else QUESTION_GEN_PROMPT_ASSOC)
            try:
                q = generator.chat([{"role": "user", "content": prompt.format(
                    tier=TIERS[tier], chunk=chunk["doc"][:2500])}]).strip()
            except Exception:
                continue
            if len(q) < 25:
                continue
            samples.append({"question": q, "reference": chunk["doc"],
                            "ground_truth": "", "source": "paper_protocol",
                            "difficulty_tier": tier, "pool": pool,
                            "source_kind": chunk["kind"],
                            "source_chunk_id": chunk["id"],
                            "source_relation": chunk["relation"]})
            tier_sizes[tier] += 1
    logger.info("Paper-protocol questions: %s (pool=%s, kinds=%s)", tier_sizes, pool,
                {k: sum(1 for s in samples if s["source_kind"] == k)
                 for k in ("pmc", "primekg")})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info("Paper-protocol questions cached to %s (resume-safe).", cache_path)
    return samples

@tracked_run(
    name="base-paper-benchmark",
    job_type="benchmark",
    config=lambda num_samples=10, output_path=None, question_pool="test": {
        "samples": num_samples, "question_pool": question_pool},
    log_results=log_base_paper_results,
)
def run_benchmark(num_samples: int = 10, output_path: Optional[Path] = None,
                  question_pool: str = "test"):
    """Execute complete base paper evaluation benchmark across the 3 systems."""
    logger.info("Initializing Base Paper Evaluator...")
    evaluator = BasePaperEvaluator()

    # Import system pipelines
    from src.baselines.baseline_llm import BaselineLLM
    from src.baselines.vanilla_vector_rag import VanillaVectorRAG
    from src.core.hcad_retriever import HCADRetriever

    logger.info("Loading baseline and HCAD models...")
    baseline_llm = BaselineLLM()
    vanilla_rag = VanillaVectorRAG()
    hcad_rag = HCADRetriever(fail_on_vector_error=True)
    logger.info("Pre-warming retriever components (graph, vector store, NER models)...")
    hcad_rag._ensure_components()
    _ = hcad_rag.graph
    _ = hcad_rag.hypergraph
    if getattr(getattr(hcad_rag, "_ner", None), "degraded", False):
        raise RuntimeError(
            "SciSpacy NER failed to load (BiomedicalNER in degraded regex "
            "mode). Refusing to run base paper benchmark on degraded features."
        )

    # Gate-mode guard: the Oct-25 run silently switched to the live TypeSafe
    # gate because a key was present in the environment, making it
    # incomparable with the local-calibrated anchor run. Fail loudly if the
    # active mode differs from the explicitly declared expectation.
    gate_mode = hcad_rag.gate_mode
    expected_gate_mode = os.environ.get("HCAD_EXPECTED_GATE_MODE", "").strip() or None
    logger.info("JEV gate mode: %s (expected: %s)", gate_mode, expected_gate_mode or "any")
    if expected_gate_mode and gate_mode != expected_gate_mode:
        raise RuntimeError(
            f"Gate mode mismatch: expected HCAD_EXPECTED_GATE_MODE="
            f"{expected_gate_mode!r}, got {gate_mode!r}. Refusing to run: "
            f"benchmark results are only comparable within one gate mode. "
            f"Unset the key, remove it from the environment, or update "
            f"HCAD_EXPECTED_GATE_MODE if this configuration change is intended."
        )

    # Paper protocol first (Sec. 3.5): LLM-generated questions with source
    # chunks as references. Generation failure is FATAL: falling back to the
    # handful of curated questions silently produces a non-protocol
    # benchmark that cannot go in the report, while the notebook cell
    # itself (! shell call) keeps running as if nothing happened.
    samples = generate_paper_protocol_questions(num_samples, pool=question_pool)
    logger.info("Using paper-protocol questions (%d).", len(samples))
    if not samples:
        # An empty sample set (e.g., the LLM server died before generation)
        # must NOT silently produce an empty results file - refuse loudly.
        raise RuntimeError(
            "No benchmark questions were generated or loaded. Refusing to "
            "write empty results. Check that the LLM provider (ollama serve "
            "on port 11434, or the configured provider) is reachable, then "
            "re-run."
        )
    logger.info(f"Loaded {len(samples)} clinical test cases.")

    # Per-sample results are appended to the checkpoint and all aggregation
    # reads back from it (see _records_from_checkpoint) - resumed runs
    # therefore aggregate over the complete sample set, not just the
    # questions evaluated in the current session.

    checkpoint_path = RESULTS_DIR / os.getenv(
        "BENCHMARK_CHECKPOINT", "base_paper_checkpoint.jsonl")
    question_cache_path = _paper_question_cache_path(num_samples, question_pool)
    question_set_id = _question_set_id(samples)
    # Off-box checkpoint upload cadence (sessions can die mid-run; the artifact
    # in W&B is the only copy that survives a killed Kaggle session).
    checkpoint_upload_every = int(os.getenv("WANDB_CHECKPOINT_EVERY", "5"))
    evaluated_this_session = 0
    completed_questions = _completed_questions_from_checkpoint(
        checkpoint_path, question_pool, question_set_id
    )
    logger.info("Resuming: %d samples already checkpointed.", len(completed_questions))

    for idx, sample in enumerate(samples):
        q = sample["question"]
        ref = sample["reference"]
        if q in completed_questions:
            logger.info("Skipping checkpointed sample %d/%d.", idx + 1, len(samples))
            continue
        logger.info(f"Evaluating Sample {idx+1}/{len(samples)}: {q[:70]}...")

        # 1. Generate responses & measure retrieval latencies
        # Baseline LLM
        resp_baseline = _validate_generated_response("Baseline LLM", baseline_llm.generate(q))
        logger.info("  Baseline LLM response (%d chars): %.120s...", len(resp_baseline), resp_baseline)

        # Vanilla RAG
        t0 = time.perf_counter()
        vanilla_contexts = vanilla_rag.retrieve(q, top_k=5)
        t_vanilla_ret = time.perf_counter() - t0
        resp_vanilla = _validate_generated_response(
            "Vanilla RAG", vanilla_rag.retrieve_and_generate(q, contexts=vanilla_contexts)
        )
        vanilla_evidence = _format_retrieved_evidence(vanilla_contexts, "Doc")  # neutral tags
        logger.info("  Vanilla RAG response (%d chars, %d contexts): %.120s...", len(resp_vanilla), len(vanilla_contexts), resp_vanilla)

        # HCAD-RAG
        t0 = time.perf_counter()
        hcad_contexts = hcad_rag.retrieve(q)
        t_hcad_ret = time.perf_counter() - t0
        logger.info("  HCAD-RAG retrieved %d contexts in %.1fs", len(hcad_contexts), t_hcad_ret)
        resp_hcad = _validate_generated_response(
            "HCAD-RAG", hcad_rag.retrieve_and_generate(q, contexts=hcad_contexts)
        )
        hcad_evidence = _format_retrieved_evidence(hcad_contexts, "Doc")  # neutral tags
        logger.info("  HCAD-RAG response (%d chars): %.120s...", len(resp_hcad), resp_hcad)

        # Judge inputs use neutral [Doc i] citation tags for every arm (blind
        # judging); the raw responses are kept for citation precision below.
        judge_baseline = _blind_citation_tags(resp_baseline)
        judge_vanilla = _blind_citation_tags(resp_vanilla)
        judge_hcad = _blind_citation_tags(resp_hcad)

        # 2. Scoring-Based Assessment (0-100)
        score_base = evaluator.evaluate_scoring(
            q, judge_baseline, ref, fallback_on_error=True
        )
        score_van = evaluator.evaluate_scoring(
            q, judge_vanilla, ref, answer_evidence=vanilla_evidence, fallback_on_error=True
        )
        score_hcad = evaluator.evaluate_scoring(
            q, judge_hcad, ref, answer_evidence=hcad_evidence, fallback_on_error=True
        )
        logger.info("  Scores - Baseline: %.1f, Vanilla: %.1f, HCAD: %.1f",
                     score_base.overall_score, score_van.overall_score, score_hcad.overall_score)

        # Robustness metrics (src/evaluation/metrics.py). Retrieval quality
        # needs gold relevance labels, which the paper protocol provides for
        # free: the source chunk that generated the question. Curated-fallback
        # samples have no retrieval gold and are excluded from NDCG/Recall.
        sample_retrieval: Dict[str, Dict[str, float]] = {}
        if sample.get("source") == "paper_protocol":
            for system, contexts in (
                ("Vanilla RAG", vanilla_contexts),
                ("HCAD-RAG", hcad_contexts),
            ):
                relevance = _ranked_relevance(ref, contexts)
                sample_retrieval[system] = {
                    "ndcg_at_k": round(ndcg_at_k(relevance, k=RETRIEVAL_K), 4),
                    "recall_at_k": round(recall_at_k(relevance, k=RETRIEVAL_K), 4),
                }

        sample_citations: Dict[str, Dict[str, Any]] = {}
        for system, response, contexts in (
            ("Vanilla RAG", resp_vanilla, vanilla_contexts),
            ("HCAD-RAG", resp_hcad, hcad_contexts),
        ):
            sample_citations[system] = citation_precision(
                response, [_context_document(c) for c in contexts]
            )

        # 3. Selection-Based Assessment (with order counterbalancing)
        # Run AB order
        sel_van_ab = evaluator.evaluate_selection(
            q, judge_hcad, judge_vanilla, ref,
            label_a="HCAD-RAG", label_b="Vanilla RAG",
            evidence_a=hcad_evidence, evidence_b=vanilla_evidence,
            fallback_on_error=True,
        )
        # Run BA order (swap answers)
        sel_van_ba = evaluator.evaluate_selection(
            q, judge_vanilla, judge_hcad, ref,
            label_a="Vanilla RAG", label_b="HCAD-RAG",
            evidence_a=vanilla_evidence, evidence_b=hcad_evidence,
            fallback_on_error=True,
        )
        # Merge: use AB result if both agree, otherwise Tie
        sel_van = _merge_selection_results(sel_van_ab, sel_van_ba, "HCAD-RAG")

        sel_base_ab = evaluator.evaluate_selection(
            q, judge_hcad, judge_baseline, ref,
            label_a="HCAD-RAG", label_b="Baseline LLM",
            evidence_a=hcad_evidence,
            fallback_on_error=True,
        )
        sel_base_ba = evaluator.evaluate_selection(
            q, judge_baseline, judge_hcad, ref,
            label_a="Baseline LLM", label_b="HCAD-RAG",
            evidence_b=hcad_evidence,
            fallback_on_error=True,
        )
        sel_base = _merge_selection_results(sel_base_ab, sel_base_ba, "HCAD-RAG")

        # Tri-arm protocol: the baseline arm must appear in every selection
        # comparison - add the vanilla-vs-baseline pair (2 extra judge calls
        # per question) so the win-rate matrix covers all three pairings.
        sel_vb_ab = evaluator.evaluate_selection(
            q, judge_vanilla, judge_baseline, ref,
            label_a="Vanilla RAG", label_b="Baseline LLM",
            evidence_a=vanilla_evidence,
            fallback_on_error=True,
        )
        sel_vb_ba = evaluator.evaluate_selection(
            q, judge_baseline, judge_vanilla, ref,
            label_a="Baseline LLM", label_b="Vanilla RAG",
            evidence_b=vanilla_evidence,
            fallback_on_error=True,
        )
        sel_vb = _merge_selection_results(sel_vb_ab, sel_vb_ba, "Vanilla RAG")

        logger.info("  Selection - HCAD vs Vanilla: %s, HCAD vs Baseline: %s, "
                     "Vanilla vs Baseline: %s",
                     sel_van.get("overall_winner"), sel_base.get("overall_winner"),
                     sel_vb.get("overall_winner"))

        # per-sample checkpoint with full provenance
        with open(checkpoint_path, "a", encoding="utf-8") as ckpt:
            ckpt.write(json.dumps({
                "question": q,
                "source": sample.get("source", "curated"),
                "difficulty_tier": sample.get("difficulty_tier", 1),
                "source_kind": sample.get("source_kind", ""),
                "source_chunk_id": sample.get("source_chunk_id", ""),
                "question_pool": sample.get("pool", question_pool),
                "question_set_id": question_set_id,
                # P2d provenance: grounded vs parametric (abstained) HCAD
                # answers, so the report can quantify abstentions without
                # reconstructing them from logs.
                "grounded_mode": "grounded" if hcad_contexts else "parametric",
                "n_hcad_contexts": len(hcad_contexts),
                "scores": {
                    "Baseline LLM": score_base.to_dict(),
                    "Vanilla RAG": score_van.to_dict(),
                    "HCAD-RAG": score_hcad.to_dict(),
                },
                "selection": {
                    "HCAD_vs_Vanilla": sel_van,
                    "HCAD_vs_Baseline": sel_base,
                    "Vanilla_vs_Baseline": sel_vb,
                },
                "latency": {
                    "Vanilla RAG": t_vanilla_ret,
                    "HCAD-RAG": t_hcad_ret,
                },
                "retrieval": sample_retrieval,
                "citations": sample_citations,
            }) + "\n")

        evaluated_this_session += 1
        if checkpoint_upload_every > 0 and evaluated_this_session % checkpoint_upload_every == 0:
            upload_checkpoint_artifact(
                [checkpoint_path, question_cache_path]
            )

    # Aggregate from the checkpoint so resumed runs include previously
    # evaluated questions of this sample set.
    records = _records_from_checkpoint(
        checkpoint_path, {s["question"] for s in samples}, question_pool=question_pool,
        question_set_id=question_set_id,
    )
    grounded_mode_counts = _grounded_mode_counts(
        checkpoint_path, question_pool, question_set_id
    )

    # Compute aggregate statistics
    def calc_mean_scores(score_records: List[Dict[str, Any]]) -> Dict[str, float]:
        if not score_records:
            return {}
        dims = ["comprehensiveness", "diversity", "empowerment", "logical", "readability", "overall_score"]
        return {dim: sum(r[dim] for r in score_records) / len(score_records) for dim in dims}

    final_scores = {sys_name: calc_mean_scores(recs) for sys_name, recs in records["scoring"].items()}
    scores_by_source = {
        source: {sys_name: calc_mean_scores(recs) for sys_name, recs in source_records.items()}
        for source, source_records in records["scoring_by_source"].items()
    }

    # Compute Win-Rates
    def calc_win_rate(win_records: List[Dict[str, Any]], target_label: str = "HCAD-RAG") -> Dict[str, Any]:
        if not win_records:
            return {}
        total = len(win_records)
        overall_wins = sum(1 for r in win_records if r["overall_winner"] == target_label)
        ties = sum(1 for r in win_records if r.get("overall_winner") == "Tie")
        losses = total - overall_wins - ties
        criteria_wins = {c: 0 for c in BasePaperEvaluator.CRITERIA_LIST}
        for r in win_records:
            cw = r.get("criteria_winners", {})
            for c in BasePaperEvaluator.CRITERIA_LIST:
                if cw.get(c) == target_label:
                    criteria_wins[c] += 1

        return {
            "target_system": target_label,
            "overall_win_rate_pct": (overall_wins / total) * 100.0,
            "tie_rate_pct": (ties / total) * 100.0,
            "decisive_win_rate_pct": (
                (overall_wins / (overall_wins + losses)) * 100.0
                if overall_wins + losses else 0.0
            ),
            "criteria_win_rates_pct": {c: (cnt / total) * 100.0 for c, cnt in criteria_wins.items()},
        }

    # Tri-arm protocol: each pairing's win rate is reported from its first
    # arm's perspective (HCAD for the two HCAD pairs, Vanilla for the
    # vanilla-vs-baseline pairing).
    selection_targets = {
        "HCAD_vs_Vanilla": "HCAD-RAG",
        "HCAD_vs_Baseline": "HCAD-RAG",
        "Vanilla_vs_Baseline": "Vanilla RAG",
    }
    final_win_rates = {
        comparison: calc_win_rate(recs, selection_targets.get(comparison, "HCAD-RAG"))
        for comparison, recs in records["selection"].items() if recs
    }
    win_rates_by_source = {
        source: {
            comparison: calc_win_rate(recs, selection_targets.get(comparison, "HCAD-RAG"))
            for comparison, recs in comparisons.items() if recs
        }
        for source, comparisons in records["selection_by_source"].items()
    }

    avg_latencies = {
        sys_name: sum(lats) / len(lats) if lats else 0.0
        for sys_name, lats in records["latency"].items()
    }
    # Tri-arm protocol: the no-retrieval baseline sits at 0.0s by definition.
    if "Baseline LLM" in final_scores:
        avg_latencies.setdefault("Baseline LLM", 0.0)

    # Robustness metric 1+2: NDCG@K / Recall@K over paper-protocol samples.
    retrieval_quality: Dict[str, Any] = {
        "k": RETRIEVAL_K,
        "n_samples": len(records["retrieval"]["HCAD-RAG"]),
        "systems": {},
    }
    for system, ret_records in records["retrieval"].items():
        retrieval_quality["systems"][system] = {
            metric: (
                round(sum(r[metric] for r in ret_records) / len(ret_records), 4)
                if ret_records
                else None
            )
            for metric in ("ndcg_at_k", "recall_at_k")
        }

    # Robustness metric 3: citation precision of generated answers.
    citation_summary: Dict[str, Any] = {}
    for system, cit_records in records["citations"].items():
        precisions = [r["precision"] for r in cit_records if r.get("precision") is not None]
        citation_summary[system] = {
            "n_answers_with_citations": len(precisions),
            "n_answers": len(cit_records),
            "total_citations": sum(r.get("n_citations", 0) for r in cit_records),
            "mean_precision": (
                round(sum(precisions) / len(precisions), 4) if precisions else None
            ),
        }

    # Tri-arm protocol: the baseline arm is the zero-citation ungrounded
    # reference point (0% citation coverage, 0 citations) - it must appear
    # in the citation comparison, not just Vanilla vs HCAD.
    if "Baseline LLM" not in citation_summary:
        n_baseline = len(records["scoring"].get("Baseline LLM", []))
        citation_summary["Baseline LLM"] = {
            "n_answers_with_citations": 0,
            "n_answers": n_baseline,
            "total_citations": 0,
            "mean_precision": None,
        }

    # Robustness metric 4: exact McNemar test on paired per-question
    # correctness (overall_score >= CORRECTNESS_THRESHOLD counts as correct).
    per_question = records["per_question"]

    def _correct_flags(system: str) -> List[bool]:
        return [
            (rec.get("scores", {}).get(system, {}).get("overall_score", 0.0) or 0.0)
            >= CORRECTNESS_THRESHOLD
            for rec in per_question
        ]

    significance_testing: Dict[str, Any] = {
        "correctness_threshold": CORRECTNESS_THRESHOLD,
        "n_paired_samples": len(per_question),
    }
    hcad_flags = _correct_flags("HCAD-RAG")
    for comparison, other_system in (
        ("HCAD_vs_Vanilla", "Vanilla RAG"),
        ("HCAD_vs_Baseline", "Baseline LLM"),
    ):
        if hcad_flags:
            significance_testing[comparison] = mcnemar_test(
                hcad_flags, _correct_flags(other_system)
            )
        else:
            significance_testing[comparison] = None

    summary_results = {
        "run_manifest": {  # provenance for every result file
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "num_samples": len(samples),
            "judge": {"provider": evaluator.judge.provider, "model": evaluator.judge.model},
            "prompt_mode": "zero_shot" if not __import__("src.config", fromlist=["ENABLE_FEW_SHOT"]).ENABLE_FEW_SHOT else "few_shot",
            "random_seed": RANDOM_SEED,
            "question_pool": question_pool,
            "question_set_id": question_set_id,
            # Gate-mode provenance: comparable only within one mode.
            "gate_mode": gate_mode,
            # Learned vs static query-adaptive tensor (ablation runs set
            # HCAD_TENSOR_MODE=learned).
            "tensor_mode": os.environ.get("HCAD_TENSOR_MODE", "static"),
        },
        "scoring_based_assessment": final_scores,
        "scoring_by_difficulty_tier": {
            str(t): {s_: (sum(r["overall_score"] for r in recs) / len(recs) if recs else 0.0)
                     for s_, recs in sys_map.items()}
            for t, sys_map in records["scoring_by_tier"].items()
        },
        "scoring_based_assessment_by_source": scores_by_source,
        "selection_based_assessment": final_win_rates,
        "selection_based_assessment_by_source": win_rates_by_source,
        "retrieval_latency_seconds": avg_latencies,
        "retrieval_quality": retrieval_quality,
        "citation_precision": citation_summary,
        "grounded_mode": grounded_mode_counts,
        "significance_testing": significance_testing,
        "num_samples": len(samples),
        "sample_source_counts": {
            source: sum(1 for sample in samples if sample.get("source", "curated") == source)
            for source in sorted({sample.get("source", "curated") for sample in samples})
        },
        "judge": {
            "provider": evaluator.judge.provider,
            "model": evaluator.judge.model,
        },
    }

    out_file = output_path or (RESULTS_DIR / os.getenv(
        "BENCHMARK_RESULTS_FILE", "base_paper_benchmark_results.json"))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    logger.info(f"Benchmark results successfully saved to {out_file}")
    return summary_results

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_benchmark(num_samples=n)
