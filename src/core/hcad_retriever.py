import logging
from typing import Any, Optional

import numpy as np

from src.config import (
    CANDIDATE_POOL_MAX,
    CHANNEL_MISSING_PRIOR,
    ENABLE_FEW_SHOT,
    ENTITY_LINK_THRESHOLD,
    FILTERING_K,
    GRAPH_TRIPLES_PER_ENTITY,
    GRAPH_TRIPLES_PER_HYPEREDGE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL_NAME,
    MAX_EVIDENCE_ITEMS,
    MIN_PHI_SCORE,
    NARY_FACTS_PER_QUERY,
    QUERY_RELATION_TO_PRIMEKG,
    SCORE_WEIGHT_ENTITY,
    SCORE_WEIGHT_MODIFIER,
    SCORE_WEIGHT_RELATION,
    aniso_dim_weighting_enabled,
    dense_weight,
    map_query_relation,
    map_relation_text,
    presence_masking_enabled,
)
from src.data.triple_text import canonical_direction, triple_to_sentence

logger = logging.getLogger(__name__)

_ENTITY_TYPE_VOCAB = ["drug", "disease", "gene", "protein", "chemical",
                       "phenotype", "pathway", "anatomy", "exposure", "other"]
_RELATION_VOCAB = ["treats", "contraindicated", "causes", "interacts",
                   "associated", "biomarker", "carries", "indication",
                   "off-label", "other"]


def _entity_type_histogram(entities: dict) -> list[float]:
    """10-dim presence histogram over NER entity categories (values in [0,1])."""
    hist = [0.0] * len(_ENTITY_TYPE_VOCAB)
    counts = []
    for key, items in (entities or {}).items():
        n = len(items) if isinstance(items, list) else 0
        if not n:
            continue
        k = str(key).lower()
        idx = next((i for i, v in enumerate(_ENTITY_TYPE_VOCAB) if v in k),
                   len(_ENTITY_TYPE_VOCAB) - 1)
        hist[idx] = max(hist[idx], 1.0)
        counts.append((idx, n))
    if counts and counts[0][1]:
        # scale by the largest per-category count, capped at 1
        top = max(n for _, n in counts)
        for idx, n in counts:
            hist[idx] = min(1.0, n / top)
    return hist


def _relation_onehot(rel_names: list[str]) -> list[float]:
    """10-dim one-hot of the query's extracted relation type."""
    vec = [0.0] * len(_RELATION_VOCAB)
    for name in rel_names:
        idx = next((i for i, v in enumerate(_RELATION_VOCAB) if v in name),
                   len(_RELATION_VOCAB) - 1)
        if any(v in name for v in _RELATION_VOCAB[:-1]):
            vec[idx] = 1.0
            return vec
    vec[-1] = 1.0
    return vec


class HCADRetriever:
    """End-to-end HCAD-RAG retrieval and generation pipeline."""

    def __init__(
        self,
        graph=None,
        collection=None,
        model_name: Optional[str] = None,
        k: float = FILTERING_K,
        fail_on_vector_error: bool = False,
    ):
        self._graph = graph
        self._collection = collection
        self.model_name = model_name
        self.k = k
        self.fail_on_vector_error = fail_on_vector_error

        self._ner: Optional[object] = None
        self._relation_extractor: Optional[object] = None
        self._decoupler: Optional[object] = None
        self._transformer: Optional[object] = None
        self._hypergraph: Optional[object] = None
        self._filter: Optional[object] = None
        self._prompt_builder: Optional[object] = None
        self._entity_linker: Optional[object] = None
        self._learned_weights: Optional[dict] = None
        self._node_lookup: Optional[dict] = None
        self._decision_gate: Optional[object] = None
        self._entity_res_cache: dict[str, Optional[str]] = {}
        # Learned query-adaptive tensor (HCAD_TENSOR_MODE=learned). The
        # override hook is used ONLY by the tensor trainer to evaluate
        # sampled weight vectors; production retrieval always predicts
        # from the tensor itself.
        self._tensor: Optional[object] = None
        self._tensor_override: Optional[tuple] = None
        self._last_tensor_inputs: Optional[tuple] = None
        self._last_channel_sims: Optional["np.ndarray"] = None
        self._last_candidate_pool: Optional[list] = None

    def _handle_vector_error(self, exc: Exception) -> None:
        logger.warning("Vector store retrieval failed: %s", exc)
        if self.fail_on_vector_error:
            raise RuntimeError("Vector store retrieval failed during benchmark retrieval.") from exc

    def _get_tensor(self):
        """Load the learned dynamic tensor when HCAD_TENSOR_MODE=learned.

        Static heuristic remains the default; a missing or unreadable
        checkpoint falls back to static with a warning (never breaks
        retrieval).
        """
        import os
        if (self._tensor is None
                and os.getenv("HCAD_TENSOR_MODE", "static") == "learned"):
            try:
                from src.core.dynamic_tensor import load_tensor
                self._tensor = load_tensor()
                logger.info("Learned dynamic tensor active (models/dynamic_tensor.pt).")
            except Exception as exc:
                logger.warning("HCAD_TENSOR_MODE=learned but no usable tensor (%s); "
                               "falling back to static weights.", exc)
        return self._tensor

    def _load_components(self):
        """Initialize components on first access."""
        from src.core.ner_extractor import BiomedicalNER, SimpleNER
        from src.core.relation_extractor import RelationExtractor
        from src.core.semantic_decoupler import SemanticDecoupler
        from src.core.anisotropic_transformer import AnisotropicTransformer
        from src.core.hypergraph_builder import HypergraphBuilder
        from src.core.statistical_filter import StatisticalFilter
        from src.core.prompt_builder import PromptBuilder
        from src.core.decision_gate import DecisionGate

        logger.info("Initializing HCAD-RAG components...")

        try:
            from src.core.ner_extractor import get_biomedical_ner
            self._ner = get_biomedical_ner()
        except Exception as e:
            logger.warning(f"BiomedicalNER unavailable ({e}), using SimpleNER fallback.")
            self._ner = SimpleNER()

        self._relation_extractor = RelationExtractor()
        self._decoupler = SemanticDecoupler()
        self._transformer = AnisotropicTransformer()

        # Hypergraph is lazily constructed via self.hypergraph
        self._filter = StatisticalFilter(k=self.k)
        self._prompt_builder = PromptBuilder()
        self._decision_gate = DecisionGate(
            min_relevance_score=0.25, max_evidence_items=MAX_EVIDENCE_ITEMS
        )

        try:
            from src.core.entity_linker import SapBERTLinker
            self._entity_linker = SapBERTLinker()
            self._entity_linker.build_index(self.graph)
        except Exception as e:
            logger.debug(f"Entity linker not initialized: {e}")
            self._entity_linker = None

        # Audit/paper-parity: n-ary fact index from PrimeKG shared-tail
        # contexts (base-paper Sec. 3.2 high-order correlations, but curated).
        try:
            from src.core.fact_hypergraph import build_fact_index_from_kg
            self._fact_index = build_fact_index_from_kg(self.graph)
            logger.info("N-ary fact index: %d clinical facts.", len(self._fact_index.facts))
        except Exception as e:
            logger.debug(f"Fact index unavailable: {e}")
            self._fact_index = None

        try:
            from src.core.weight_learner import load_learned_weights
            self._learned_weights = load_learned_weights()
            if self._learned_weights and "filter_k" in self._learned_weights:
                self._filter = StatisticalFilter(k=self._learned_weights["filter_k"])
        except Exception:
            self._learned_weights = None

    @property
    def graph(self):
        if self._graph is None:
            from src.data.build_knowledge_graph import load_knowledge_graph
            self._graph = load_knowledge_graph()
        return self._graph

    @property
    def hypergraph(self):
        if self._hypergraph is None:
            from src.core.hypergraph_builder import HypergraphBuilder
            raw_graph = getattr(self.graph, "graph", self.graph)
            if raw_graph is not None:
                if hasattr(self.graph, "name_to_node") and not hasattr(raw_graph, "name_to_node"):
                    try:
                        raw_graph.name_to_node = self.graph.name_to_node
                    except Exception:
                        pass
                self._hypergraph = HypergraphBuilder(raw_graph)
        return self._hypergraph

    @property
    def gate_mode(self) -> str:
        """Active JEV gate mode: 'live-typesafe' or 'local-calibrated'.

        Exposed for benchmark provenance: the mode is recorded in every
        run_manifest and fail-loud compared against HCAD_EXPECTED_GATE_MODE
        so a stray TYPESAFE_API_KEY cannot silently switch the measurement
        chain between runs.
        """
        if self._decision_gate is None:
            return "uninitialized"
        return getattr(self._decision_gate, "mode", "unknown")

    @property
    def collection(self):
        if self._collection is None:
            from src.data.build_vector_store import load_vector_store
            self._collection = load_vector_store()
        return self._collection

    def _ensure_components(self):
        if self._ner is None:
            self._load_components()

    # Entity Resolution

    def _resolve_entity_in_graph(self, entity_name: str) -> Optional[str]:
        """Resolve an entity name to a canonical PrimeKG node name with in-memory caching."""
        if entity_name in self._entity_res_cache:
            return self._entity_res_cache[entity_name]

        # Exact (case-insensitive) PrimeKG name first: precise and cheap.
        # SapBERT nearest-neighbour linking previously ran first at a 0.50
        # cosine floor, which re-mapped names that exist verbatim in the
        # graph onto near neighbours and expanded the wrong entity.
        res = self._find_entity_in_graph(entity_name)
        if res is None and entity_name and entity_name.strip():
            res = self._find_entity_in_graph(entity_name.lower())

        if res is None and self._entity_linker is not None:
            try:
                matches = self._entity_linker.link(
                    entity_name, top_k=1, threshold=ENTITY_LINK_THRESHOLD)
                if matches:
                    res = matches[0]["canonical_name"]
                    logger.debug(
                        "SapBERT linked '%s' -> '%s' (score=%.3f)",
                        entity_name, res, matches[0]["score"],
                    )
            except Exception as e:
                logger.warning(f"SapBERT linking failed for '{entity_name}': {e}")

        self._entity_res_cache[entity_name] = res
        return res

    def _find_entity_in_graph(self, entity_name: str) -> Optional[str]:
        """Find the best matching node name in the graph (fast O(1) hash lookup)."""
        if not entity_name or not isinstance(entity_name, str):
            return None

        try:
            if hasattr(self.graph, "get_node_by_name"):
                try:
                    result = self.graph.get_node_by_name(entity_name)
                    if result:
                        return entity_name
                except KeyError:
                    pass

            graph_obj = getattr(self.graph, "graph", self.graph)
            if graph_obj is None:
                return None

            if entity_name in graph_obj:
                return entity_name

            # Build lazy lowercase lookup index once
            if self._node_lookup is None:
                logger.info("Building fast O(1) PrimeKG node name hash index...")
                name_to_node = getattr(self.graph, "name_to_node", {})
                if name_to_node:
                    self._node_lookup = {k.lower(): k for k in name_to_node.keys()}
                else:
                    # Fallback: extract names from node attributes
                    if graph_obj is not None and hasattr(graph_obj, "nodes"):
                        self._node_lookup = {
                            str(graph_obj.nodes[n].get("name", "")).lower(): str(graph_obj.nodes[n].get("name", ""))
                            for n in graph_obj.nodes
                            if graph_obj.nodes[n].get("name")
                        }
                if self._node_lookup is None:
                    self._node_lookup = {}
                logger.info(f"Node name hash index ready with {len(self._node_lookup)} entries.")

            entity_lower = entity_name.lower()
            if self._node_lookup and entity_lower in self._node_lookup:
                return self._node_lookup[entity_lower]

        except Exception as e:
            logger.debug(f"Entity lookup failed for '{entity_name}': {e}")

        return None

    @staticmethod
    def _kg_candidate(head: str, head_type: str, relation: str, tail: str,
                      tail_type: str, score: float, source: str,
                      display_relation: Optional[str] = None) -> dict:
        """A KG-triple candidate rendered exactly like the vector store.

        The edge is first oriented canonically (PrimeKG stores both
        directions), so "X is indicated for Y" from the vector store and
        from graph expansion produce the SAME document and dedup.
        """
        head, head_type, tail, tail_type, _ = canonical_direction(
            head, head_type, relation, tail, tail_type)
        return {
            "document": triple_to_sentence(head, relation, tail),
            "metadata": {
                "head_name": head,
                "tail_name": tail,
                "relation": relation,
                "display_relation": display_relation or relation,
                "head_type": head_type,
                "tail_type": tail_type,
            },
            "score": score,
            "source": source,
        }

    @staticmethod
    def _query_primekg_relations(relations: Optional[list]) -> set:
        """PrimeKG relation types the query's rule relations ask about."""
        wanted: set = set()
        for rel in relations or []:
            if isinstance(rel, dict):
                wanted |= QUERY_RELATION_TO_PRIMEKG.get(
                    str(rel.get("relation", "")).upper(), set())
        return wanted

    def _get_candidate_contexts(
        self, query: str, entities: dict, n_vector: int = 30,
        relations: Optional[list] = None,
    ) -> list[dict]:
        """Candidate pool from three channels, deduplicated by document text.

        1. dense vector search over the whole store (same 30-NN Vanilla
           draws its top-5 from);
        2. relation-typed hyperedges around each linked query entity,
           emitting the concrete facts each hyperedge contains, relations
           the query asks about first;
        3. n-ary PrimeKG facts in which a query entity is a member.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        seen_docs: set = set()
        self._last_query_embedding = None

        # -- 1. dense channel ------------------------------------------------
        vector_cands: list[dict] = []
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            emb_model = (
                self._decoupler._model
                if hasattr(self, "_decoupler")
                and self._decoupler
                and hasattr(self._decoupler, "_model")
                and self._decoupler._model
                else SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            )
            # Chroma's collection uses cosine distance. Keep the query vector
            # on the same unit-normalized scale as the indexed vectors.
            q_vec = np.asarray(emb_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            ), dtype=np.float32)
            self._last_query_embedding = q_vec[0]
            results = self.collection.query(
                query_embeddings=q_vec.tolist(), n_results=n_vector
            )
            documents = (results.get("documents") or [[]])[0] or []
            metadatas = (results.get("metadatas") or [[]])[0] or []
            distances = (results.get("distances") or [[]])[0] or []
            for i, doc in enumerate(documents):
                meta = dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {}
                cand = {
                    "document": doc,
                    "metadata": meta,
                    "score": 1.0 - distances[i] if i < len(distances) else 0.0,
                    "source": "vector",
                }
                rel = meta.get("relation") or ""
                if rel and meta.get("head_name") and meta.get("tail_name"):
                    oriented = self._kg_candidate(
                        meta["head_name"], meta.get("head_type", ""), rel,
                        meta["tail_name"], meta.get("tail_type", ""),
                        cand["score"], "vector", meta.get("display_relation"),
                    )
                    if oriented["document"] != doc:
                        # reversed PrimeKG copy: re-rendered, so its stored
                        # similarity no longer describes this text
                        cand = oriented
                    else:
                        cand["dense_score"] = cand["score"]
                else:
                    cand["dense_score"] = cand["score"]
                if cand["document"] not in seen_docs:
                    vector_cands.append(cand)
                    seen_docs.add(cand["document"])
        except Exception as e:
            self._handle_vector_error(e)

        # -- entity resolution (shared by channels 2 and 3) -------------------
        name_to_node = getattr(self.graph, "name_to_node", {}) or {}
        graph_nx = getattr(self.graph, "graph", self.graph)
        if graph_nx is None or not hasattr(graph_nx, "nodes"):
            import networkx as nx
            graph_nx = nx.MultiDiGraph()

        raw_names = [
            ent["text"] for ent_list in (entities or {}).values() for ent in ent_list
            if isinstance(ent, dict) and ent.get("text")
        ]
        # deterministic: first mention order, de-duplicated, capped at 5
        raw_names = list(dict.fromkeys(raw_names))[:5]
        resolved: list[tuple[str, Any]] = []   # (canonical name, node id)
        for entity_name in raw_names:
            matched_name = self._resolve_entity_in_graph(entity_name)
            if not matched_name:
                continue
            node_id = name_to_node.get(matched_name, matched_name)
            if node_id is not None and node_id in graph_nx:
                resolved.append((matched_name, node_id))

        # -- 2. hyperedge channel --------------------------------------------
        graph_cands: list[dict] = []
        hypergraph = self.hypergraph
        wanted_relations = self._query_primekg_relations(relations)

        def _node(n):
            data = graph_nx.nodes[n] if n in graph_nx else {}
            return data.get("name", str(n)), data.get("node_type", "unknown")

        if hypergraph is not None:
            for matched_name, node_id in resolved:
                try:
                    hyperedges = hypergraph.relation_hyperedges(node_id)
                except Exception as e:
                    logger.debug("Hyperedge expansion failed for '%s': %s", matched_name, e)
                    continue
                if not hyperedges:
                    continue
                # relations the query asks about first, then importance
                hyperedges.sort(key=lambda h: (h["relation"] not in wanted_relations,
                                               -h["importance"]))
                max_importance = max(h["importance"] for h in hyperedges) or 1.0
                emitted = 0
                for he in hyperedges:
                    if emitted >= GRAPH_TRIPLES_PER_ENTITY:
                        break
                    per_edge = 0
                    for u, v in he["edges"]:
                        if per_edge >= GRAPH_TRIPLES_PER_HYPEREDGE or emitted >= GRAPH_TRIPLES_PER_ENTITY:
                            break
                        head_name, head_type = _node(u)
                        tail_name, tail_type = _node(v)
                        cand = self._kg_candidate(
                            head_name, head_type, he["relation"], tail_name, tail_type,
                            0.5 * he["importance"] / max_importance, "graph_hyperedge",
                        )
                        if cand["document"] in seen_docs:
                            continue
                        graph_cands.append(cand)
                        seen_docs.add(cand["document"])
                        per_edge += 1
                        emitted += 1

        # -- 3. n-ary fact channel --------------------------------------------
        # Paper parity (Sec. 3.4): retrieval against hyperedges/facts in
        # addition to entity->vertex retrieval. Anchors are the linked
        # canonical PrimeKG names (plus raw mentions), so facts are found
        # by the same names the graph uses.
        fact_cands: list[dict] = []
        fact_index = getattr(self, "_fact_index", None)
        if fact_index is not None:
            anchors = {name.lower() for name, _ in resolved}
            anchors |= {name.lower() for name in raw_names}
            if anchors:
                for f in fact_index.retrieve(anchors, top_k=NARY_FACTS_PER_QUERY):
                    doc = fact_index.render(f, focus=anchors)
                    if doc in seen_docs:
                        continue
                    heads = fact_index.displayed_heads(f, focus=anchors)
                    fact_cands.append({
                        "document": doc,
                        "metadata": {
                            "head_name": heads[0] if heads else "",
                            "tail_name": f.roles.get("anchor", ""),
                            "relation": f.predicate,
                            "display_relation": f.predicate,
                            "fact_arity": len(f.roles),
                            "provenance": f.source_id,
                        },
                        "score": 0.5 * f.confidence,
                        "source": "nary_fact",
                    })
                    seen_docs.add(doc)

        # Dense hits first, then facts, then graph triples fill the rest of
        # the pool (graph triples are the most numerous channel).
        candidates = vector_cands + fact_cands + graph_cands
        return candidates[:CANDIDATE_POOL_MAX]

    def _dense_similarities(self, query: str, candidates: list[dict]) -> np.ndarray:
        """Full-text cosine(query, candidate document) for every candidate.

        Vector hits reuse Chroma's similarity; graph/fact candidates (and
        re-oriented triples) are encoded with the same embedding model,
        served from the shared embedding cache on repeats.
        """
        q = getattr(self, "_last_query_embedding", None)
        if q is None:
            q = self._decoupler.encode_texts([query])[0]
        q = np.asarray(q, dtype=np.float32)
        dense = np.zeros(len(candidates), dtype=np.float32)
        missing = [i for i, c in enumerate(candidates) if "dense_score" not in c]
        for i, c in enumerate(candidates):
            if "dense_score" in c:
                dense[i] = float(c["dense_score"])
        if missing:
            vecs = self._decoupler.encode_texts(
                [str(candidates[i].get("document") or "") for i in missing])
            dense[missing] = vecs @ q
        return dense

    def retrieve(
        self,
        query: str,
        k: Optional[float] = None,
        n_candidates: int = 30,
    ) -> list[dict]:
        """Retrieve and rank candidate contexts using decoupled structural similarity and adaptive filtering."""
        self._ensure_components()
        k = k if k is not None else self.k

        # Per-query diagnostics are reset up front so an early return (no
        # candidates) can never leave the previous query's pool/sims behind
        # for the trainer or the dev-pool harness to misread.
        self._last_tensor_inputs = None
        self._last_channel_sims = None
        self._last_masked_sims = None
        self._last_candidate_pool = None
        self._last_phi_scores = None
        self._last_dense_sims = None
        self._last_dense_weight = 0.0

        entities = self._ner.extract_entities(query)
        relations = self._relation_extractor.extract_rules(query, entities)
        # Query relation labels ("TREATS") rendered with the same canonical
        # verb phrases the candidates' relation channel uses.
        query_relations = [
            {**r, "relation": map_query_relation(str(r.get("relation", "")))}
            for r in (relations or []) if isinstance(r, dict)
        ]

        Q, query_presence = self._decoupler.decouple(
            query, entities=entities, relations=query_relations, return_presence=True
        )

        all_entity_names = [
            e.get("text", "") for ents in entities.values() for e in ents if isinstance(e, dict)
        ]
        # Filter out empty strings and ensure we have valid entity names
        all_entity_names = [name for name in all_entity_names if name and isinstance(name, str)]
        primary_entity = all_entity_names[0] if all_entity_names else ""

        tensor = self._get_tensor()
        lam_override = a_override = None
        if self._tensor_override is not None:
            lam_override, a_override = self._tensor_override
        if tensor is not None or lam_override is not None:
            import torch
            from src.core.dynamic_tensor import make_query_features
            deg, conf, idf = self._transformer.graph_block_stats(
                self.graph, primary_entity)
            rel_names = [str(r.get("relation", "")).lower() for r in (relations or [])
                         if isinstance(r, dict)]
            neg_flag = 1.0 if any("contra" in r or "not_" in r for r in rel_names) else 0.0
            type_hist = _entity_type_histogram(entities)
            rel_onehot = _relation_onehot(rel_names)
            feats = make_query_features(
                None,
                max(0.0, min(1.0, deg)),
                max(0.0, min(1.0, conf)),
                max(0.0, min(1.0, idf)),
                neg_flag,
                0.0,
                type_hist,
                rel_onehot,
            )
            Q_t = torch.tensor(np.asarray(Q, dtype=np.float32))
            if lam_override is not None:
                lam_t, a_t = lam_override, a_override
            else:
                lam_t, a_t = tensor(Q_t, feats)
            # Always record (Q, feats) for this query - the listwise trainer
            # keys its channel-weight adapter to these inputs even when the
            # warm-up pass ran under an explicit override.
            self._last_tensor_inputs = (Q_t.detach(), feats.detach())
            weights = lam_t.detach().cpu().numpy().astype(np.float64)
            component_a = a_t.detach().cpu().numpy()
        else:
            if aniso_dim_weighting_enabled():
                weights = self._transformer.compute_weights(self.graph, primary_entity)
            else:
                # isotropic: weighted cosine reduces to plain cosine
                dim = int(np.asarray(Q).shape[-1])
                weights = np.full(dim, 1.0 / dim, dtype=np.float32)
            component_a = None

        candidates = self._get_candidate_contexts(
            query, entities, n_vector=n_candidates, relations=relations
        )
        if not candidates:
            return []

        # Vectorized batch decoupling of all candidates in a single tensor operation
        candidate_docs = []
        entities_list = []
        relations_list = []
        for ctx in candidates:
            ctx_doc = ctx.get("document", "")
            ctx_meta = ctx.get("metadata") or {}
            ctx_entity = ctx_meta.get("head_name", "")
            # Deliverable-B relation-vocabulary mapping: raw PrimeKG schema
            # labels ("drug protein") barely cosine-match natural query verbs
            # ("targets"); map to canonical clinical verb phrases.
            ctx_relation = map_relation_text(ctx_meta.get("relation") or "")
            ctx_tail = ctx_meta.get("tail_name", "")

            candidate_docs.append(ctx_doc)
            ctx_entities_dict = None
            ctx_relations_list = None
            if ctx_entity:
                ctx_entities_dict = {"Entity": [
                    {"text": ctx_entity, "start": 0, "end": len(ctx_entity)},
                ]}
                if ctx_tail:
                    ctx_entities_dict["Entity"].append(
                        {"text": ctx_tail, "start": len(ctx_entity)+1,
                         "end": len(ctx_entity)+1+len(ctx_tail)}
                    )
            if ctx_relation:
                ctx_relations_list = [{"relation": ctx_relation}]

            entities_list.append(ctx_entities_dict)
            relations_list.append(ctx_relations_list)

        C_batch, candidate_presence = self._decoupler.decouple_batch(
            candidate_docs,
            entities_list=entities_list,
            relations_list=relations_list,
            as_numpy=True,
            return_presence=True,
        )
        if component_a is not None:
            channel_weights = (float(component_a[0]), float(component_a[1]),
                               float(component_a[2]))
        else:
            channel_weights = (SCORE_WEIGHT_ENTITY, SCORE_WEIGHT_RELATION,
                               SCORE_WEIGHT_MODIFIER)

        if presence_masking_enabled():
            # Deliverable-B: query-conditioned channel-presence masking.
            phi_scores, masked_sims = self._transformer.compute_masked_similarities(
                Q, C_batch, weights,
                query_presence=query_presence,
                candidate_presence=candidate_presence,
                alpha=channel_weights[0], beta=channel_weights[1],
                gamma=channel_weights[2],
                missing_prior=CHANNEL_MISSING_PRIOR,
            )
            self._last_masked_sims = masked_sims
        else:
            phi_scores = self._transformer.compute_batch_component_similarities(
                Q, C_batch, weights,
                alpha=channel_weights[0], beta=channel_weights[1],
                gamma=channel_weights[2],
            )

        # Hybrid score: the decoupled channels compare short extracted
        # fragments, so on their own they discard the full-text semantic
        # match. Phi = w_d * dense + (1 - w_d) * Phi_struct keeps the dense
        # ranking as a floor and lets the E/R/M channels re-rank on top.
        phi_struct = np.asarray(phi_scores, dtype=np.float32)
        w_d = dense_weight()
        if w_d > 0.0:
            dense = self._dense_similarities(query, candidates)
        else:
            dense = np.zeros(len(candidates), dtype=np.float32)
        phi_scores = (w_d * dense + (1.0 - w_d) * phi_struct).astype(np.float32)
        for ctx, s_struct, s_dense in zip(candidates, phi_struct, dense):
            ctx["phi_struct"] = float(s_struct)
            ctx["dense_score"] = float(s_dense)
        self._last_dense_sims = dense
        self._last_dense_weight = w_d

        # Per-channel similarities for this (query, candidate pool)
        # pair: the listwise trainer learns the query-adaptive channel
        # weights over exactly these columns. The PRE-GATE candidate
        # pool is stashed too, index-aligned with the sims rows - the
        # trainer's listwise loss runs over the full pool (gold may be
        # present but gated out, and the loss must be able to teach the
        # channel weights to rank it back in).
        self._last_channel_sims = self._transformer.compute_channel_similarities(
            Q, C_batch, weights)
        self._last_candidate_pool = list(candidates)
        # Pre-gate provenance for the dev-pool harness (tau_sat calibration
        # and paired before/after retrieval metrics).
        self._last_phi_scores = np.asarray(phi_scores)
        self._last_query_presence = query_presence
        self._last_candidate_presence = candidate_presence

        filtered = self._filter.filter_contexts(
            candidates,
            np.asarray(phi_scores),
            k=k,
            max_top_k=MAX_EVIDENCE_ITEMS,
        )

        # the previous "Fix #2" re-admitted candidates
        # BELOW the absolute floor Phi_min = 0.10 whenever both gates rejected
        # everything - defeating Gate 1 exactly on out-of-domain queries.
        # Correct behavior: abstain (empty context) so the generator relies on
        # parametric knowledge rather than being distracted by noise.
        if not filtered:
            logger.warning(
                "All %d candidates rejected by dual-gating (floor %.2f). "
                "Abstaining from retrieval - parametric mode.",
                len(candidates), MIN_PHI_SCORE,
            )
            return []

        # JEV-inspired Decision Gate: Verify relevance & prune token bloat
        if self._decision_gate is not None:
            filtered = self._decision_gate.verify_and_prune_candidates(filtered, query)

        return filtered

    def retrieve_and_generate(
        self,
        query: str,
        k: Optional[float] = None,
        model: Optional[str] = None,
        contexts: Optional[list[dict]] = None,
    ) -> str:
        from src.core.llm_client import UnifiedLLMClient

        self._ensure_components()
        model = model or self.model_name
        if contexts is None:
            contexts = self.retrieve(query, k=k)
        messages = self._prompt_builder.build_rag_prompt(query, contexts)

        try:
            client = UnifiedLLMClient(model=model, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
            return client.chat(messages)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Error: LLM generation failed - {e}]"

    def answer_mcq(
        self,
        question: str,
        options: dict,
        k: Optional[float] = None,
        model: Optional[str] = None,
        n_samples: int = 1,
    ) -> tuple[str, str]:
        """Answer a multiple-choice question using retrieved context."""
        from collections import Counter
        from src.core.llm_client import UnifiedLLMClient
        from src.baselines.baseline_llm import extract_answer_letter

        self._ensure_components()
        model = model or self.model_name
        contexts = self.retrieve(question, k=k)
        messages = self._prompt_builder.build_mcq_prompt(
            question, options, contexts=contexts, few_shot=ENABLE_FEW_SHOT,
        )

        try:
            client = UnifiedLLMClient(model=model, temperature=LLM_TEMPERATURE, max_tokens=512)
            if n_samples > 1:
                votes = []
                responses = []
                for i in range(n_samples):
                    text = client.chat(messages, temperature=0.3, max_tokens=768)
                    letter = extract_answer_letter(text)
                    votes.append(letter)
                    responses.append(f"Sample {i+1} ({letter}):\n{text}")

                # deterministic tie-break - first sample wins.
                counts = Counter(votes)
                majority_vote = max(votes, key=lambda v: (counts[v], -votes.index(v)))
                full_response = "\n\n".join(responses)
                return majority_vote, full_response

            full_response = client.chat(messages)
            return extract_answer_letter(full_response), full_response
        except Exception as e:
            logger.error(f"MCQ answer failed: {e}")
            return "X", f"[Error: {e}]"
