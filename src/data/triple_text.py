"""Single source of truth for rendering PrimeKG triples as evidence text.

The vector store, the graph-expansion path and the retriever must render a
triple identically: text-level dedup across retrieval channels and the
benchmark's gold-relevance matching both compare document strings.
Previously the vector store wrote "X is indicated for Y." while graph
expansion wrote "X indication Y.", so the same fact entered the candidate
pool twice and graph evidence never matched a gold chunk.

Kept free of heavy imports (no chromadb / torch) so every layer can use it.
"""

from __future__ import annotations

from src.config import RELATION_HEAD_TYPE

# Readable verb phrase per PrimeKG relation. Must stay byte-identical to
# what the built vector store contains (existing stores are not rebuilt).
TRIPLE_REL_MAP = {
    "drug_protein": "targets protein",
    "carries": "carries gene",
    "cause": "causes",
    "contraindication": "is contraindicated in",
    "indication": "is indicated for",
    "off-label use": "has off-label use in",
    "drug_drug": "interacts with drug",
    "phenotype_protein": "is associated with protein",
    "protein_protein": "interacts with protein",
    "disease_phenotype_positive": "presents with phenotype",
    "disease_phenotype_negative": "lacks phenotype",
    "disease_protein": "is associated with protein",
    "biomarker": "is a biomarker for",
    "disease_disease": "is related to disease",
    "pathway_protein": "involves protein",
}


def triple_to_sentence(head: str, relation: str, tail: str) -> str:
    """Convert a KG triple to a readable natural language sentence."""
    readable_rel = TRIPLE_REL_MAP.get(relation, relation.replace("_", " "))
    return f"{head} {readable_rel} {tail}."


def canonical_direction(
    head: str, head_type: str, relation: str, tail: str, tail_type: str,
) -> tuple[str, str, str, str, bool]:
    """Orient a PrimeKG edge so its head has the relation's head type.

    PrimeKG lists every edge in both directions; the reversed copy reads
    wrongly ("Type 2 diabetes is indicated for Metformin"). Returns
    (head, head_type, tail, tail_type, swapped).
    """
    want = RELATION_HEAD_TYPE.get((relation or "").strip().lower())
    ht = (head_type or "").strip().lower()
    tt = (tail_type or "").strip().lower()
    if want and ht != want and tt == want:
        return tail, tail_type, head, head_type, True
    return head, head_type, tail, tail_type, False
