import logging
from typing import Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EVIDENCE_WORD_BUDGET,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    VANILLA_TOP_K,
    VECTORDB_DIR,
)
from src.baselines.baseline_llm import BaselineLLM, _extract_answer_letter

logger = logging.getLogger(__name__)

class VanillaVectorRAG:
    """Conventional Vector-RAG baseline using flat Top-K cosine retrieval.

    Uses S-PubMedBert-MS-MARCO to encode queries into a single d-dimensional
    vector and retrieves the top-K most similar chunks from a ChromaDB store.
    Retrieved contexts are concatenated and passed to an LLM for answer
    generation.

    This baseline highlights the core limitation addressed by HCAD-RAG:
    a single-vector similarity score cannot distinguish whether a retrieved
    passage matches the query's *entity*, *relation*, or *modifier* intent.

    Attributes:
        embedding_model: SentenceTransformer encoder.
        collection: ChromaDB collection handle.
        top_k: Number of passages to retrieve per query.
        llm: BaselineLLM instance used for answer generation.
    """

    def __init__(
        self,
        top_k: int = VANILLA_TOP_K,
        model: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        """Initialize the Vanilla Vector-RAG baseline.

        Loads the embedding model and connects to the persistent ChromaDB
        vector store. Falls back to creating the collection if it does not
        exist (it will be empty until data ingestion runs).

        Args:
            top_k: Number of Top-K results to retrieve (default: config.VANILLA_TOP_K).
            model: Ollama model name for generation (default: config.LLM_MODEL).
            temperature: LLM sampling temperature.
            max_tokens: LLM max generation tokens.
        """
        self.top_k = top_k

        # ── Embedding Model ──────────────────────────────────────────────
        logger.info(
            "Loading embedding model: %s (d=%d)",
            EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
        )
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device, local_files_only=True)
        except Exception:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device)

        # ── ChromaDB ─────────────────────────────────────────────────────
        persist_dir = str(VECTORDB_DIR)
        logger.info("Connecting to ChromaDB at %s", persist_dir)
        self.client = chromadb.PersistentClient(path=persist_dir)

        try:
            self.collection = self.client.get_collection(
                name=CHROMA_COLLECTION_NAME,
            )
            count = self.collection.count()
            logger.info(
                "Loaded collection '%s' with %d documents",
                CHROMA_COLLECTION_NAME, count,
            )
        except Exception:
            # fail loudly. Previously a missing store
            # silently degraded this baseline into a zero-shot LLM, corrupting
            # any benchmark that assumed real retrieval.
            raise RuntimeError(
                f"Vector store collection '{CHROMA_COLLECTION_NAME}' not found or empty. "
                "Run `python src/main.py prepare` before evaluating Vanilla RAG."
            )

        # ── LLM for generation ───────────────────────────────────────────
        self.llm = BaselineLLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(
            "VanillaVectorRAG initialized - top_k=%d, collection=%s",
            self.top_k, CHROMA_COLLECTION_NAME,
        )

    def _encode_query(self, query: str) -> list[float]:
        """Encode a text query into a d-dimensional embedding vector.

        Args:
            query: Natural-language query string.

        Returns:
            List of floats representing the query embedding.
        """
        embedding = self.embedding_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True,
        )
        return embedding.tolist()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve top-K passages by cosine similarity.

        Demonstrates the flat-embedding retrieval approach:
        ``Dim(S_base) = 1 × d``  - a single similarity score with no
        semantic decomposition.

        Args:
            query: Natural-language query.
            top_k: Override for the number of results (default: self.top_k).

        Returns:
            List of dicts, each containing:
              - "id": Document ID in ChromaDB
              - "text": Passage text
              - "score": Cosine similarity score (1 - distance)
              - "metadata": Any stored metadata dict
        """
        # validate top_k; `or` treated 0 as "use default".
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        k = top_k if top_k is not None else self.top_k

        if self.collection.count() == 0:
            logger.warning("ChromaDB collection is empty - no results to return")
            return []

        query_embedding = self._encode_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB returns cosine *distance*; convert to similarity
            score = 1.0 - dist
            retrieved.append({
                "id": doc_id,
                "text": text,
                "score": round(float(score), 6),
                "metadata": meta or {},
            })

        logger.debug(
            "VanillaVectorRAG.retrieve - query: %.60s... → %d results "
            "(top score=%.4f)",
            query, len(retrieved),
            retrieved[0]["score"] if retrieved else 0.0,
        )
        return retrieved

    def _build_rag_prompt(
        self,
        query: str,
        contexts: list[dict],
    ) -> str:
        """Build the RAG prompt by concatenating retrieved contexts.

        Args:
            query: Original user query.
            contexts: List of retrieved passage dicts.

        Returns:
            Formatted prompt string with contexts + query.
        """
        if not contexts:
            return query

        # Fairness: both systems operate under the same evidence word budget
        # (EVIDENCE_WORD_BUDGET). Without this, vanilla's generator receives
        # the full text of its top-K passages (thousands of words for PMC
        # chunks) while HCAD's decision gate enforces a structured token
        # economy, so answer-length differences would reflect the evidence
        # BUDGET rather than evidence SELECTION - the actual hypothesis.
        # Clipping follows the same sentence-boundary rule as the benchmark's
        # judge-side formatter (mid-sentence clipping can sever negations).
        context_parts = []
        remaining_words = EVIDENCE_WORD_BUDGET
        for i, ctx in enumerate(contexts, 1):
            if remaining_words <= 0:
                break
            words = str(ctx.get("text", "")).split()
            if not words:
                continue
            budget_text = " ".join(words[:remaining_words])
            cut = max(budget_text.rfind(". "), budget_text.rfind("! "),
                      budget_text.rfind("? "))
            selected = budget_text if cut == -1 else budget_text[: cut + 1]
            context_parts.append(
                f"[Context {i}] (similarity={ctx['score']:.4f})\n{selected.strip()}"
            )
            remaining_words -= len(selected.split())
        context_block = "\n\n---\n\n".join(context_parts)

        from src.core.prompt_builder import synthesis_instructions

        # Identical synthesis instructions to HCAD-RAG and the baseline; the
        # arms differ only in the evidence supplied.
        prompt = (
            f"Clinical Query:\n{query}\n\n"
            f"Retrieved Contexts:\n{context_block}\n\n"
            f"{synthesis_instructions('Context')}"
        )
        return prompt

    def retrieve_and_generate(
        self,
        query: str,
        contexts: Optional[list[dict]] = None,
    ) -> str:
        """Full RAG pipeline: retrieve contexts then generate an answer.

        Args:
            query: Medical question in natural language.
            contexts: Optional pre-retrieved contexts.

        Returns:
            LLM-generated answer string.
        """
        if contexts is None:
            contexts = self.retrieve(query)
        prompt = self._build_rag_prompt(query, contexts)

        from src.core.llm_client import UnifiedLLMClient
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            client = UnifiedLLMClient(
                model=self.llm.model,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
            )
            response = client.chat(messages)
        except Exception:
            response = self.llm._call_ollama(prompt, system=LLM_SYSTEM_PROMPT)

        logger.debug(
            "VanillaVectorRAG.retrieve_and_generate - "
            "%d contexts, response_preview=%.80s...",
            len(contexts), response,
        )
        return response
