"""Blueprint 5.2 - Native n-ary clinical facts with provenance and polarity.

Clinical assertions are typed facts with named roles, predicate, polarity,
qualifiers, confidence, and provenance - never pairwise collapses of
neighbor co-membership. Traversal is a two-step walk over the bipartite
entity/fact incidence graph (hypergraph PageRank on H H^T).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

# PrimeKG relations whose semantics carry no direction: both endpoints are
# co-equal association members (disease_disease is symmetric).
UNDIRECTED_RELATIONS = {"disease_disease"}

# Relations whose PrimeKG name already encodes absence; rendering them with
# an extra negation cue would produce double negation.
NEGATED_RELATIONS = {"disease_phenotype_negative"}

@dataclass(frozen=True)
class ClinicalFact:
    predicate: str
    roles: dict                     # {"drug_a": ..., "disease": ..., "phenotype": ...}
    polarity: str = "asserts"       # "asserts" | "negated" | "uncertain"
    qualifiers: dict = field(default_factory=dict)
    confidence: float = 1.0
    source_id: str = ""
    direction: str = "directed"     # "directed" | "undirected"
    # Every head sharing this (tail, relation) context, up to an indexing
    # cap. `roles` holds only the displayed subset (max_arity); indexing
    # on all members means a query entity is found even when it is not
    # among the first few displayed heads.
    members: tuple = ()

    @property
    def head_names(self) -> list:
        """Role values excluding the anchor (tail) - the head side of the fact."""
        return [v for k, v in self.roles.items() if k != "anchor"]

    @property
    def tail_name(self):
        """The anchor role - the tail side of a directed fact."""
        return self.roles.get("anchor")

class FactIndex:
    def __init__(self) -> None:
        self.facts: list[ClinicalFact] = []
        self.by_entity: dict[str, list[int]] = defaultdict(list)

    def add(self, fact: ClinicalFact) -> None:
        idx = len(self.facts)
        self.facts.append(fact)
        for e in set(fact.roles.values()) | set(fact.members):
            self.by_entity[e].append(idx)
            low = str(e).lower()
            if low not in self.by_entity:
                self.by_entity[low] = self.by_entity[e]

    def retrieve(self, anchors: set[str], predicate: str | None = None, top_k: int = 20):
        """Score facts by role coverage x confidence (no co-membership fabrication)."""
        hits: dict[int, float] = defaultdict(float)
        for e in anchors:
            key = str(e)
            rows = self.by_entity.get(key) or self.by_entity.get(key.lower()) or ()
            for i in rows:
                f = self.facts[i]
                if predicate is not None and f.predicate != predicate:
                    continue
                roles_l = {str(r).lower() for r in f.roles.values()}
                roles_l |= {str(m).lower() for m in f.members}
                overlap = len(roles_l & {str(a).lower() for a in anchors}) / max(len(f.roles), 1)
                hits[i] += f.confidence * (0.5 + 0.5 * overlap)
        ranked = sorted(hits.items(), key=lambda kv: -kv[1])[:top_k]
        return [self.facts[i] for i, _ in ranked]

    def expand(self, anchors: set[str], hops: int = 2, max_facts: int = 100):
        """Two-step hypergraph walk; preserves fact boundaries and provenance."""
        seen, frontier, out = set(anchors), set(anchors), []
        for _ in range(hops):
            nxt: set[str] = set()
            for f in self.retrieve(frontier):
                out.append(f)
                nxt.update(f.roles.values())
                if len(out) >= max_facts:
                    return out
            frontier = nxt - seen
            seen |= frontier
            if not frontier:
                break
        return out

    @staticmethod
    def displayed_heads(fact: ClinicalFact, focus=None) -> list:
        """Heads to display, with any focus (query) entities listed first.

        A focus entity that is a member of the fact but not among the
        displayed subset replaces the last displayed heads, so the
        rendered evidence always names the entity the query asked about.
        """
        heads = list(fact.head_names)
        if not focus:
            return heads
        focus_l = {str(f).lower() for f in focus}
        anchor = str(fact.tail_name or "").lower()
        pool = list(dict.fromkeys(heads + [m for m in fact.members if m not in heads]))
        hits = [m for m in pool if str(m).lower() in focus_l and str(m).lower() != anchor]
        if not hits:
            return heads
        rest = [h for h in heads if h not in hits]
        return (hits + rest)[: max(len(heads), len(hits))]

    def render(self, fact: ClinicalFact, focus=None) -> str:
        """Faithful, role-directed evidence rendering.

        Directed facts render as ``heads --[predicate]--> tail`` (asymmetric);
        undirected facts (disease_disease) as ``heads --[predicate]-- tail``.
        Negated relations (e.g. disease_phenotype_negative) render with exactly
        ONE negation token - their PrimeKG name already encodes absence, so a
        second cue would produce double negation ("does not lack").
        """
        heads = ", ".join(self.displayed_heads(fact, focus))
        tail = fact.tail_name or "UNKNOWN"

        if fact.polarity == "negated":
            rendered = f"{heads} are NOT associated with {tail} (negated relation: {fact.predicate})."
        elif fact.direction == "undirected":
            rendered = f"{heads} --[{fact.predicate}]-- {tail} (undirected)."
        elif fact.polarity == "uncertain":
            rendered = f"{heads} --[{fact.predicate}]--> {tail} (uncertain)."
        else:
            rendered = f"{heads} --[{fact.predicate}]--> {tail}."

        self._single_negation_guard(rendered, fact)
        return rendered

    @staticmethod
    def _single_negation_guard(rendered: str, fact: ClinicalFact) -> None:
        """A negated fact must not compose with a second negation token.

        Catches composition double negation ("does not lack", "not without",
        "no absence of") while ignoring entity names that legitimately contain
        words like "absence of". Raises ValueError on violation.
        """
        low = rendered.lower()
        doubled = re.search(
            r"\b(not|no|never)\s+(not|no|never|lack of|absence of|without)\b",
            low,
        )
        if doubled:
            raise ValueError(
                f"Double-negation guard violated: negated relation rendered as "
                f"{doubled.group()!r} in {rendered!r}"
            )

# Clinical relations are indexed first so that a memory cap can never
# starve them in favour of molecular-only contexts.
RELATION_PRIORITY = [
    "indication", "contraindication", "off-label use", "drug_effect",
    "disease_phenotype_positive", "disease_phenotype_negative",
    "disease_protein", "drug_protein", "biomarker", "disease_disease",
    "phenotype_protein", "protein_protein",
]


def build_fact_index_from_kg(kg, max_arity: int = 6, min_members: int = 2,
                             max_facts_per_relation: int = 15_000,
                             max_index_members: int = 30,
                             relation_whitelist=None) -> "FactIndex":
    """Mine n-ary clinical facts from PrimeKG shared-tail contexts.

    Base-paper parity : their high-order
    correlations are LLM-mined association sets (~4k from NeurologyCorp).
    PrimeKG gives strictly better facts for free: every (tail, relation)
    context groups all heads that share it, e.g. every drug contraindicated
    in disease C becomes ONE n-ary fact with provenance.

    Coverage: the previous builder stopped after the first 10,000 contexts
    in edge-iteration order (an arbitrary subset) and indexed only the 6
    lowest-id heads. Contexts are now taken relation by relation in
    clinical priority order with a per-relation cap, and every member (up
    to max_index_members) is indexed while max_arity heads are displayed.
    """
    idx = FactIndex()
    # Default to clinically relevant relations only (same policy as the
    # vector store) - indexing all ~8M edges (incl. DDI) OOMs the process.
    if relation_whitelist is None:
        relation_whitelist = set(RELATION_PRIORITY)
    by_context = {}
    for u, v, d in kg.graph.edges(data=True):
        rel = d.get("relation", "unknown")
        if rel not in relation_whitelist:
            continue
        by_context.setdefault((v, rel), []).append(u)

    def _name(node):
        return kg.graph.nodes[node].get("name", str(node)) if node in kg.graph else str(node)

    ordered_relations = [r for r in RELATION_PRIORITY if r in relation_whitelist]
    ordered_relations += sorted(set(relation_whitelist) - set(ordered_relations))
    by_relation: dict = {}
    for (tail, rel), heads in by_context.items():
        by_relation.setdefault(rel, []).append((tail, heads))

    for rel in ordered_relations:
        n_rel = 0
        for tail, heads in by_relation.get(rel, []):
            unique_heads = sorted(set(heads))
            if len(unique_heads) < min_members:
                continue
            tail_name = _name(tail)
            head_names = [_name(h) for h in unique_heads[: max_arity]]
            members = tuple(_name(h) for h in unique_heads[: max_index_members])
            roles = {"member_%d" % i: n for i, n in enumerate(head_names)}
            roles["anchor"] = tail_name
            idx.add(ClinicalFact(
                predicate=rel,
                roles=roles,
                polarity="negated" if rel in NEGATED_RELATIONS else "asserts",
                confidence=0.9,
                source_id="primekg",
                direction="undirected" if rel in UNDIRECTED_RELATIONS else "directed",
                members=members,
            ))
            n_rel += 1
            if n_rel >= max_facts_per_relation:
                break
    return idx
