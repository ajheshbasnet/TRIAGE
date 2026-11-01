"""Retrieval-quality fixes: hybrid scoring, real hyperedge triples,
canonical triple orientation, word-boundary keywords, score-ordered gate.

All tests run without model downloads: the embedding model, NER and the
vector store are replaced by deterministic stubs.
"""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Triple rendering / orientation
# ---------------------------------------------------------------------------

def test_reversed_primekg_edge_is_reoriented():
    from src.data.triple_text import canonical_direction, triple_to_sentence

    head, _, tail, _, swapped = canonical_direction(
        "Type 2 diabetes", "disease", "indication", "Metformin", "drug")
    assert swapped
    assert triple_to_sentence(head, "indication", tail) == \
        "Metformin is indicated for Type 2 diabetes."

    # already canonical, and symmetric relations are left alone
    assert not canonical_direction("Metformin", "drug", "indication",
                                   "Type 2 diabetes", "disease")[4]
    assert not canonical_direction("A", "disease", "disease_disease", "B", "disease")[4]


def test_vector_store_and_retriever_share_one_renderer():
    from src.data.build_vector_store import _triple_to_sentence
    from src.data.triple_text import triple_to_sentence

    assert _triple_to_sentence is triple_to_sentence


# ---------------------------------------------------------------------------
# Relation-typed hyperedges emit their own facts
# ---------------------------------------------------------------------------

def _drug_graph():
    g = nx.MultiDiGraph()
    g.add_node(1, name="Metformin", node_type="drug")
    g.add_node(2, name="Type 2 diabetes", node_type="disease")
    g.add_node(3, name="PRKAA1", node_type="gene/protein")
    g.add_node(4, name="Renal failure", node_type="disease")
    # PrimeKG lists every edge in both directions
    for u, v, rel in [(1, 2, "indication"), (1, 3, "drug_protein"),
                      (1, 4, "contraindication")]:
        g.add_edge(u, v, relation=rel)
        g.add_edge(v, u, relation=rel)
    return g


def test_relation_hyperedges_carry_their_own_edges(monkeypatch):
    from src.core.hypergraph_builder import HypergraphBuilder

    monkeypatch.delenv("HCAD_EXPANSION_RULE", raising=False)
    hyperedges = HypergraphBuilder(_drug_graph()).relation_hyperedges(1)
    by_rel = {h["relation"]: h for h in hyperedges}
    assert set(by_rel) == {"indication", "drug_protein", "contraindication"}
    for he in hyperedges:
        assert he["nodes"][0] == 1
        assert len(he["edges"]) == len(he["nodes"]) - 1
        for (u, v), member in zip(he["edges"], he["nodes"][1:]):
            assert member in (u, v) and 1 in (u, v)
    assert by_rel["contraindication"]["nodes"][1:] == [4]


# ---------------------------------------------------------------------------
# Word-boundary keyword matching
# ---------------------------------------------------------------------------

def test_modifier_scan_is_word_bounded():
    from src.config import NEUTRAL_MODIFIER_TEXT
    from src.core.semantic_decoupler import SemanticDecoupler

    # "carcinoma" contains "no", "NOTCH1" contains "not": neither is a cue
    text = "NOTCH1 is associated with protein in renal carcinoma."
    m = SemanticDecoupler._extract_modifier_text(
        text, {"Gene": [{"text": "NOTCH1"}], "Disease": [{"text": "renal carcinoma"}]}, None)
    assert "no" not in m.split() and "not" not in m.split()

    m2 = SemanticDecoupler._extract_modifier_text(
        "Metformin is not recommended in severe renal failure.", None, None)
    assert "not" in m2.split() and "severe" in m2.split()
    assert m2 != NEUTRAL_MODIFIER_TEXT


def test_relation_extractor_modifiers_are_word_bounded():
    from src.core.relation_extractor import RelationExtractor

    assert "acute" not in RelationExtractor.extract_modifier("subacute thyroiditis", {})
    assert "acute" in RelationExtractor.extract_modifier("acute kidney injury", {})
    assert "post-" in RelationExtractor.extract_modifier("post-operative care", {})


# ---------------------------------------------------------------------------
# DecisionGate keeps relevance order
# ---------------------------------------------------------------------------

def test_gate_keeps_prose_in_relevance_order():
    from src.core.decision_gate import DecisionGate

    gate = DecisionGate(use_jev=False, max_evidence_items=8, token_budget_words=350)
    prose = {"document": "Metformin reduces hepatic glucose output in type 2 diabetes.",
             "metadata": {"relation": ""}, "phi_final": 0.90}
    triples = [{"document": f"Drug {i} targets protein P{i}.",
                "metadata": {"head_name": f"Drug {i}", "relation": "drug_protein",
                             "tail_name": f"P{i}"},
                "phi_final": 0.80 - i / 100} for i in range(5)]
    kept = gate.verify_and_prune_candidates([prose] + triples, "q")
    assert kept[0] is prose, "the most relevant item (prose) must stay first"
    rels = [c["metadata"]["relation"] for c in kept]
    # per-relation soft cap: over-cap triples are only admitted after others
    assert rels[1:4] == ["drug_protein"] * 3


def test_gate_drops_duplicate_documents():
    from src.core.decision_gate import DecisionGate

    gate = DecisionGate(use_jev=False)
    doc = {"document": "Metformin is indicated for Type 2 diabetes.",
           "metadata": {}, "phi_final": 0.9}
    kept = gate.verify_and_prune_candidates([doc, dict(doc, phi_final=0.8)], "q")
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# End-to-end retrieve() with stubs: hybrid score + dedup + orientation
# ---------------------------------------------------------------------------

DIM = 16


def _unit(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class _FakeDecoupler:
    """Structural channels identical for every text (no signal), so any
    ranking difference must come from the dense half of the score."""

    def __init__(self, dense_vectors, query_vec):
        self._model = SimpleNamespace(encode=lambda texts, **_: np.stack([query_vec]))
        self._dense = dense_vectors
        self._base = np.stack([_unit(1), _unit(2), _unit(3)])

    def decouple(self, text, entities=None, relations=None, return_presence=False, **_):
        return self._base.copy(), np.array([True, True, True])

    def decouple_batch(self, texts, entities_list=None, relations_list=None,
                       as_numpy=True, return_presence=False):
        n = len(texts)
        return np.repeat(self._base[None], n, axis=0), np.ones((n, 3), dtype=bool)

    def encode_texts(self, texts):
        return np.stack([self._dense.get(t, _unit(99)) for t in texts])


class _FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def query(self, query_embeddings, n_results):
        rows = self.rows[:n_results]
        return {"documents": [[r[0] for r in rows]],
                "metadatas": [[r[1] for r in rows]],
                "distances": [[r[2] for r in rows]]}


def _make_retriever(monkeypatch, gold_dense):
    from src.core.anisotropic_transformer import AnisotropicTransformer
    from src.core.decision_gate import DecisionGate
    from src.core.hcad_retriever import HCADRetriever
    from src.core.relation_extractor import RelationExtractor
    from src.core.statistical_filter import StatisticalFilter

    monkeypatch.delenv("HCAD_DENSE_WEIGHT", raising=False)
    monkeypatch.delenv("HCAD_ANISO_DIM_WEIGHTING", raising=False)
    monkeypatch.delenv("HCAD_TAU_SAT", raising=False)

    g = _drug_graph()
    kg = SimpleNamespace(graph=g, name_to_node={
        g.nodes[n]["name"]: n for n in g.nodes})
    prose = "Metformin lowers hepatic glucose production in type 2 diabetes."
    reversed_meta = {"head_name": "Type 2 diabetes", "head_type": "disease",
                     "relation": "indication", "display_relation": "indication",
                     "tail_name": "Metformin", "tail_type": "drug"}
    collection = _FakeCollection([
        (prose, {"doc_type": "clinical_text", "source": "pmc", "relation": ""}, 0.20),
        ("Type 2 diabetes is indicated for Metformin.", reversed_meta, 0.30),
    ])

    q = _unit(7)
    gold = "Metformin is contraindicated in Renal failure."
    # gold contraindication triple is the closest text to the query:
    # unit vector with cosine exactly gold_dense to q
    ortho = _unit(8) - float(_unit(8) @ q) * q
    ortho /= np.linalg.norm(ortho)
    dense_vectors = {gold: gold_dense * q + np.sqrt(1 - gold_dense ** 2) * ortho}

    r = HCADRetriever(graph=kg, collection=collection)
    r._ner = SimpleNamespace(extract_entities=lambda text: {
        "Drug": [{"text": "Metformin", "start": text.find("Metformin"),
                  "end": text.find("Metformin") + 9}]})
    r._relation_extractor = RelationExtractor()
    r._decoupler = _FakeDecoupler(dense_vectors, q)
    r._transformer = AnisotropicTransformer(dim=DIM)
    r._filter = StatisticalFilter(k=0.0)
    r._decision_gate = DecisionGate(use_jev=False, min_relevance_score=0.0)
    r._entity_linker = None
    r._fact_index = None
    return r, gold


def test_retrieve_reorients_and_dedups_kg_triples(monkeypatch):
    r, _ = _make_retriever(monkeypatch, gold_dense=0.95)
    r.retrieve("Is Metformin contraindicated in renal failure?")
    docs = [c["document"] for c in r._last_candidate_pool]
    assert "Type 2 diabetes is indicated for Metformin." not in docs
    assert docs.count("Metformin is indicated for Type 2 diabetes.") == 1, \
        "vector hit and graph triple must collapse into one candidate"
    graph_docs = [c["document"] for c in r._last_candidate_pool
                  if c["source"] == "graph_hyperedge"]
    assert graph_docs[0] == "Metformin is contraindicated in Renal failure.", \
        "the relation the query asks about must be expanded first"


def test_hybrid_score_surfaces_densely_relevant_candidate(monkeypatch):
    r, gold = _make_retriever(monkeypatch, gold_dense=0.95)
    out = r.retrieve("Is Metformin contraindicated in renal failure?")
    assert out and out[0]["document"] == gold
    assert out[0]["dense_score"] == pytest.approx(0.95, abs=1e-4)
    assert r._last_dense_weight > 0


def test_dense_weight_zero_restores_structure_only_score(monkeypatch):
    r, _ = _make_retriever(monkeypatch, gold_dense=0.95)
    monkeypatch.setenv("HCAD_DENSE_WEIGHT", "0")
    r.retrieve("Is Metformin contraindicated in renal failure?")
    phi = r._last_phi_scores
    # identical structural channels -> identical structure-only scores
    assert np.allclose(phi, phi[0])
