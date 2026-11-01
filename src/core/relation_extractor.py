"""
Hybrid Relation Extraction Module.

Implements two complementary strategies for extracting typed relations
between biomedical entities :

  1. **Rule-based** - keyword/pattern matching with synonym lists for
     each relation type (TREATS, CONTRAINDICATED, INHIBITS, …).
  2. **LLM-based** - uses Ollama to classify ambiguous relations when
     rule-based matching is inconclusive.

Also provides:
  - Negation detection (critical for modifier component & Context
    Distraction mitigation).
  - Modifier extraction (severity, temporal, negation qualifiers).

Usage:
    from src.core.ner_extractor import BiomedicalNER
    ner = BiomedicalNER()
    entities = ner.extract_entities("Metformin does not treat lung cancer.")
    rex = RelationExtractor()
    relations = rex.extract_rules("Metformin does not treat lung cancer.", entities)
"""

from __future__ import annotations

import json
import logging
import re
from itertools import product
from typing import Any

from src.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    NEGATION_CUES,
    RELATION_TYPES,
)

logger = logging.getLogger(__name__)

# Type aliases
EntityDict = dict[str, list[dict[str, Any]]]
RelationDict = dict[str, Any]  # single relation record

# Keyword / synonym lists per relation type (5-10 variants each)
RELATION_PATTERNS: dict[str, list[str]] = {
    "TREATS": [
        "treats", "is used to treat", "therapeutic for", "is indicated for",
        "used in the treatment of", "manages", "alleviates", "ameliorates",
        "is effective against", "provides relief for",
    ],
    "CAUSES": [
        "causes", "leads to", "results in", "induces", "triggers",
        "gives rise to", "is associated with increased risk of",
        "precipitates", "provokes",
    ],
    "CONTRAINDICATED": [
        "contraindicated", "should not be used with", "is contraindicated in",
        "must not be given", "is incompatible with", "avoid in",
        "not recommended for", "prohibited in", "unsafe for",
    ],
    "INTERACTS_WITH": [
        "interacts with", "drug interaction", "has interaction with",
        "may interact with", "combined with", "co-administered with",
        "concomitant use", "synergistic with",
    ],
    "TARGETS": [
        "targets", "binds to", "acts on", "has affinity for",
        "selectively targets", "receptor for", "ligand for",
        "modulates", "is a substrate of",
    ],
    "ASSOCIATED_WITH": [
        "associated with", "linked to", "correlated with",
        "co-occurs with", "related to", "implicated in",
        "observed in", "concurrent with",
    ],
    "INHIBITS": [
        "inhibits", "blocks", "suppresses", "antagonises", "antagonizes",
        "down-regulates", "decreases activity of", "is an inhibitor of",
        "attenuates", "represses",
    ],
    "ACTIVATES": [
        "activates", "stimulates", "enhances", "promotes",
        "up-regulates", "induces expression of", "potentiates",
        "agonises", "agonizes", "increases activity of",
    ],
    "UPREGULATES": [
        "upregulates", "up-regulates", "increases expression of",
        "elevates levels of", "overexpresses", "amplifies",
    ],
    "DOWNREGULATES": [
        "downregulates", "down-regulates", "decreases expression of",
        "reduces levels of", "silences", "suppresses expression of",
    ],
    "BIOMARKER_FOR": [
        "biomarker for", "is a marker of", "predictive of",
        "diagnostic indicator of", "prognostic marker for",
        "screening marker for",
    ],
    "NOT_RECOMMENDED": [
        "not recommended", "should be avoided", "is discouraged",
        "not advisable", "inadvisable", "generally not used",
    ],
    "INDICATED_FOR": [
        "indicated for", "approved for", "has indication for",
        "recommended for", "prescribed for", "first-line for",
        "standard of care for",
    ],
}

# Pre-compile relation patterns (case-insensitive)
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    rel: [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]
    for rel, keywords in RELATION_PATTERNS.items()
}

# Modifier keyword lists
_SEVERITY_KEYWORDS: list[str] = [
    "severe", "mild", "moderate", "acute", "chronic", "advanced",
    "early-stage", "late-stage", "refractory", "resistant", "progressive",
    "terminal", "critical", "benign", "malignant", "metastatic",
]

_TEMPORAL_KEYWORDS: list[str] = [
    "during", "after", "before", "following", "prior to", "upon",
    "within", "initially", "subsequently", "eventually", "post-",
    "pre-", "peri-", "concomitantly", "simultaneously",
]


def _keyword_pattern(kw: str) -> re.Pattern:
    """Whole-word match; prefix cues ("post-") only need a leading boundary."""
    if kw.endswith("-"):
        return re.compile(rf"\b{re.escape(kw)}", re.IGNORECASE)
    return re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)


# Substring tests made "acute" fire on "subacute" and "upon"/"within"
# fire inside longer words; same word-boundary rule as negation cues.
_SEVERITY_PATTERNS = [(kw, _keyword_pattern(kw)) for kw in _SEVERITY_KEYWORDS]
_TEMPORAL_PATTERNS = [(kw, _keyword_pattern(kw)) for kw in _TEMPORAL_KEYWORDS]

# RelationExtractor
class RelationExtractor:
    """Hybrid relation extraction with rule-based and LLM strategies.

    Parameters
    ----------
    use_llm_fallback : bool
        Whether to attempt LLM-based extraction when rules are ambiguous.
    """

    def __init__(self, use_llm_fallback: bool = False) -> None:
        self.use_llm_fallback = use_llm_fallback

    # Negation detection
    # Word-boundary cue patterns, longest-first so multi-word cues win.
    _CUE_PATTERNS: list[re.Pattern] = [
        re.compile(rf"\b{re.escape(cue)}\b", re.IGNORECASE)
        for cue in sorted(NEGATION_CUES, key=len, reverse=True)
    ]

    # Clause boundaries terminate a negation cue's scope: a cue may not cross
    # a sentence/clause separator or a contrastive conjunction to negate a
    # relation stated in a different clause.
    _CLAUSE_BOUNDARY_RE = re.compile(
        r"[.;,:?!\n]"
        r"|\b(?:however|but|although|though|whereas|while|yet|nevertheless|"
        r"nonetheless|conversely|in contrast|on the contrary)\b",
        re.IGNORECASE,
    )

    @classmethod
    def detect_negation(cls, text: str) -> tuple[bool, str]:
        """Identify negation cues in *text* with word boundaries.

        Audit fix (item 21): boundary-less substring matching made "not"
        match "noted"/"normal" and "no" match "node"/"notice", silently
        converting therapeutic indications into contraindications. Cues
        are now matched on word boundaries only.
        """
        for pat in cls._CUE_PATTERNS:
            m = pat.search(text or "")
            if m:
                return True, m.group().lower()
        return False, ""

    @classmethod
    def _same_clause(cls, segment: str) -> bool:
        """True when no clause boundary separates *segment*'s endpoints."""
        return cls._CLAUSE_BOUNDARY_RE.search(segment) is None

    @classmethod
    def detect_negation_in_clause(
        cls, text: str, head_start: int, head_end: int,
        tail_start: int, lookback: int = 40,
    ) -> tuple[bool, str]:
        """Negation cue detection bounded by clause boundaries (P2c).

        A cue negates the head->tail relation only if no clause boundary
        (``[.;,:?!\n]`` or a contrastive conjunction) separates the cue from
        the head or the tail entity. This stops cues from a previous
        sentence ("X treats Y. However, aspirin should not be used with
        Z.") from negating relations stated in a later clause.
        """
        if head_start < 0 or tail_start < 0:
            return cls.detect_negation(text)

        window_start = max(0, head_start - lookback)
        window = text[window_start:tail_start]
        head_pos = head_start - window_start
        head_end_pos = head_end - window_start

        for pat in cls._CUE_PATTERNS:
            for m in pat.finditer(window):
                if m.start() <= head_pos:
                    # cue before (or inside) the head entity
                    segment = window[m.end():head_pos]
                    if cls._same_clause(segment):
                        return True, m.group().lower()
                else:
                    # cue after the head: must share a clause with the tail
                    # span so cues in a discarded clause do not leak in.
                    tail_pos = tail_start - window_start
                    segment = window[m.end():tail_pos] if m.end() <= tail_pos \
                        else window[tail_pos:m.start()]
                    if cls._same_clause(segment):
                        return True, m.group().lower()
        return False, ""

    # Modifier extraction
    @staticmethod
    def extract_modifier(text: str, entities: EntityDict) -> str:
        """Extract qualifying modifier text that is NOT entity/relation.

        Identifies severity, temporal, and negation qualifiers remaining
        after entity and relation spans are removed.

        Returns
        -------
        str
            The concatenated modifier tokens, or ``"neutral"`` if none
            are found.
        """
        modifiers: list[str] = []

        # Severity and temporal qualifiers (whole words only)
        for kw, pat in _SEVERITY_PATTERNS + _TEMPORAL_PATTERNS:
            if pat.search(text or ""):
                modifiers.append(kw)

        # Negation
        is_negated, neg_cue = RelationExtractor.detect_negation(text)
        if is_negated:
            modifiers.append(neg_cue)

        return " ".join(modifiers) if modifiers else "neutral"

    # Rule-based extraction
    def extract_rules(
        self, text: str, entities: EntityDict
    ) -> list[RelationDict]:
        """Extract relations via keyword/pattern matching.

        For every ordered pair of entities found in *text*, check
        whether any relation keyword appears between (or near) them.

        Returns
        -------
        list[dict]
            Each dict: ``{head, head_type, relation, tail, tail_type,
            is_negated, modifier}``.
        """
        # Flatten entities into a single list with types
        flat_entities: list[dict[str, Any]] = []
        for etype, elist in entities.items():
            for ent in elist:
                if isinstance(ent, dict):
                    # Ensure entity has required fields
                    entity_copy = ent.copy()
                    entity_copy["entity_type"] = etype
                    entity_copy.setdefault("text", "")
                    # entities without real offsets used
                    # start=0/end=len(text), which zeroes out pair logic.
                    # Locate the actual mention instead.
                    if "start" not in entity_copy or entity_copy.get("start") is None:
                        ent_text = entity_copy.get("text", "")
                        found = text.find(ent_text) if ent_text else -1
                        if found == -1 and ent_text:
                            found = text.lower().find(ent_text.lower())
                        if found >= 0:
                            entity_copy["start"] = found
                            entity_copy["end"] = found + len(ent_text)
                        else:
                            entity_copy["start"] = None
                            entity_copy["end"] = None
                    elif "end" not in entity_copy or entity_copy.get("end") is None:
                        start_pos = entity_copy.get("start", 0) or 0
                        entity_copy["end"] = start_pos + len(entity_copy.get("text", ""))
                    flat_entities.append(entity_copy)

        if len(flat_entities) < 2:
            # Need at least two entities for a relation
            return self._single_entity_fallback(text, flat_entities)

        # Sort by position (entities with known offsets first)
        flat_entities.sort(key=lambda e: (e.get("start") is None, e.get("start") or 0))

        relations: list[RelationDict] = []
        seen: set[tuple[str, str, str]] = set()

        for head, tail in product(flat_entities, repeat=2):
            if head is tail:
                continue
            head_start = head.get("start")
            tail_start = tail.get("start")

            if head_start is not None and tail_start is not None:
                # Only consider pairs where head precedes tail in text
                if head_start >= tail_start:
                    continue
                between_start = head.get("end", head_start)
                between_end = tail.get("start", tail_start)
                between_text = text[between_start:between_end]
            else:
                # Mock or unaligned entities lacking in-text offsets
                head_start = -1
                tail_start = -1
                between_start = 0
                between_end = len(text)
                between_text = text

            for rel_type, patterns in _COMPILED_PATTERNS.items():
                for pat in patterns:
                    if pat.search(between_text):
                        head_text = head.get("text", "")
                        tail_text = tail.get("text", "")
                        tail_end = tail.get("end", tail_start)
                        key = (head_text, rel_type, tail_text)
                        if key not in seen:
                            seen.add(key)
                            # Scope: a small window before the head entity
                            # plus the pair span, so cues such as
                            # "No evidence that X causes Y" are caught.
                            # P2c: the cue window terminates at clause
                            # boundaries ([.;,:] and contrastive
                            # conjunctions) - a cue from a previous
                            # sentence/clause no longer negates the
                            # relation stated here.
                            is_neg, neg_cue = self.detect_negation_in_clause(
                                text,
                                head_start,
                                head.get("end", head_start),
                                tail_start,
                            )
                            modifier_span = (
                                text[head_start:tail_end]
                                if (head_start >= 0 and tail_end >= 0)
                                else text
                            )
                            modifier = self.extract_modifier(
                                modifier_span, entities
                            )
                            relations.append({
                                "head": head_text,
                                "head_type": head.get("entity_type", "Unknown"),
                                "relation": rel_type,
                                "tail": tail_text,
                                "tail_type": tail.get("entity_type", "Unknown"),
                                "is_negated": is_neg,
                                "modifier": modifier,
                            })
                        break  # one match per relation type is enough

        logger.debug("Rule-based extraction found %d relations.", len(relations))
        return relations

    def _single_entity_fallback(
        self, text: str, flat_entities: list[dict[str, Any]]
    ) -> list[RelationDict]:
        """When only 0-1 entities exist, try to infer a self-relation."""
        if not flat_entities:
            return []
        ent = flat_entities[0]
        ent_text = ent.get("text", "")
        # Check if any relation keyword appears at all
        for rel_type, patterns in _COMPILED_PATTERNS.items():
            for pat in patterns:
                if pat.search(text):
                    is_neg, _ = self.detect_negation(text)
                    modifier = self.extract_modifier(text, {})
                    return [{
                        "head": ent_text,
                        "head_type": ent.get("entity_type", "Unknown"),
                        "relation": rel_type,
                        "tail": "UNKNOWN",
                        "tail_type": "Unknown",
                        "is_negated": is_neg,
                        "modifier": modifier,
                    }]
        return []

    # LLM-based extraction (Ollama fallback)
    def extract_llm(
        self, text: str, entities: EntityDict
    ) -> list[RelationDict]:
        """Use an Ollama LLM to classify relations when rules fail.

        Constructs a structured prompt listing the entities and asking
        the model to identify relation types from the predefined set.

        Returns
        -------
        list[dict]
            Same structure as :meth:`extract_rules`.
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests library not available for LLM extraction.")
            return []

        # Build entity list string
        entity_strs: list[str] = []
        for etype, elist in entities.items():
            for ent in elist:
                entity_strs.append(f"{ent['text']} ({etype})")

        if not entity_strs:
            return []

        valid_rels = ", ".join(RELATION_TYPES)
        prompt = (
            f"Given the following biomedical text and entities, identify "
            f"all relations between entity pairs.\n\n"
            f"Text: \"{text}\"\n\n"
            f"Entities: {', '.join(entity_strs)}\n\n"
            f"Valid relation types: {valid_rels}\n\n"
            f"Return a JSON list of objects with keys: "
            f"head, head_type, relation, tail, tail_type.\n"
            f"Return ONLY the JSON list, no other text."
        )

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "temperature": LLM_TEMPERATURE,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json().get("response", "")

            # Parse JSON from the response
            # Try to find a JSON array in the response
            json_match = re.search(r"\[.*\]", result, re.DOTALL)
            if json_match:
                raw_relations = json.loads(json_match.group())
            else:
                logger.warning("LLM response did not contain valid JSON array.")
                return []

            # Enrich with negation / modifier
            relations: list[RelationDict] = []
            for raw in raw_relations:
                if not isinstance(raw, dict):
                    continue
                rel_type = raw.get("relation", "").upper()
                if rel_type not in RELATION_TYPES:
                    continue
                is_neg, _ = self.detect_negation(text)
                modifier = self.extract_modifier(text, entities)
                relations.append({
                    "head": raw.get("head", ""),
                    "head_type": raw.get("head_type", "Unknown"),
                    "relation": rel_type,
                    "tail": raw.get("tail", ""),
                    "tail_type": raw.get("tail_type", "Unknown"),
                    "is_negated": is_neg,
                    "modifier": modifier,
                })
            logger.debug("LLM extraction found %d relations.", len(relations))
            return relations

        except Exception as exc:
            logger.warning("LLM relation extraction failed: %s", exc)
            return []

    def extract_keywords(self, text: str) -> list[str]:
        """Correlation keywords (X_cor) for fact/hyperedge retrieval.

        Paper parity : the query yields an entity
        keyword set AND a correlation keyword set; the latter targets
        hyperedges. Here we surface the matched relation predicates.
        """
        text_lower = (text or "").lower()
        found = []
        for rel_type, patterns in _COMPILED_PATTERNS.items():
            for pat in patterns:
                if pat.search(text_lower):
                    found.append(rel_type.lower())
                    break
        return found

    # Unified extraction interface
    def extract(self, text: str, entities: EntityDict) -> list[RelationDict]:
        """Extract relations using rules first, with optional LLM fallback.

        Parameters
        ----------
        text : str
            Input biomedical text.
        entities : dict
            Output of :meth:`BiomedicalNER.extract_entities`.

        Returns
        -------
        list[dict]
            Extracted relations, each with ``head``, ``head_type``,
            ``relation``, ``tail``, ``tail_type``, ``is_negated``,
            ``modifier``.
        """
        relations = self.extract_rules(text, entities)

        if not relations and self.use_llm_fallback:
            logger.info("No rule-based relations found; trying LLM fallback.")
            relations = self.extract_llm(text, entities)

        # no fabricated relations. If rules and the LLM
        # both find nothing, the relation channel is *absent* - the decoupler
        # substitutes a fixed neutral anchor instead of a false association.
        return relations
