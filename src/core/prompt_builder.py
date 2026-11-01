from typing import Any, Dict, Optional

from src.config import FEW_SHOT_COUNT, LLM_SYSTEM_PROMPT


def context_document(context: Dict[str, Any]) -> str:
    """Readable text of a retrieved context (mirrors the evidence formatter)."""
    document = context.get("document") or context.get("text") or ""
    if not document:
        metadata = context.get("metadata") or {}
        head = metadata.get("head_name", "")
        relation = metadata.get("display_relation") or metadata.get("relation", "")
        tail = metadata.get("tail_name", "")
        document = f"{head} {relation} {tail}".strip()
    return str(document)


_SYNTHESIS_COVERAGE = (
    "Cover the relevant pathophysiology, first- and second-line management, "
    "contraindications and safety considerations, monitoring, differential "
    "diagnosis, and complications when the question calls for them; omit "
    "tangential details. Avoid speculation and unsupported patient-specific "
    "recommendations."
)

_SYNTHESIS_LENGTH = (
    "IMPORTANT: Keep your response focused and concise (under 800 words total). "
    "Prioritize clinical precision over exhaustive enumeration. Avoid repeating "
    "the same point across sections."
)

_SYNTHESIS_STRUCTURE = (
    "Structure your response clearly with the following sections:\n"
    "1. Executive Clinical Summary: Direct, conclusive answer to the clinical question.\n"
    "2. Pathophysiological & Evidence Analysis: In-depth clinical reasoning, citing "
    "supporting evidence where it directly confirms key facts.\n"
    "3. Clinical Management & Risk Stratification: Practical implications, differential "
    "considerations, risk factors, and management guidelines drawn from established "
    "clinical practice.\n"
    "4. Conclusion & Prognostic Outlook: Concise synthesis emphasizing patient outcomes "
    "and best practices."
)


def synthesis_instructions(evidence_tag: Optional[str] = None) -> str:
    """Answer-synthesis instructions shared by ALL benchmark arms.

    Baseline, Vanilla RAG and HCAD-RAG receive identical instructions and
    differ only in the evidence they are given, so judge-score differences
    reflect evidence selection rather than prompt wording. `evidence_tag`
    is the citation tag the arm's evidence block uses ("Evidence",
    "Context"); None means no retrieved evidence (baseline / abstention).
    """
    if evidence_tag:
        grounding = (
            "Use the relevant retrieved evidence as the factual foundation, then "
            "synthesize it with established clinical knowledge to answer the full "
            "question. Do not let gaps in the retrieved evidence make the answer "
            "incomplete. Support claims with evidence tags "
            f"(e.g. [{evidence_tag} N]) only when the cited item directly supports "
            "the claim, and make clear which points rely on established clinical "
            "knowledge instead. Never imply that a retrieved item supports details "
            "it does not contain."
        )
    else:
        grounding = (
            "No retrieved evidence is available. Draw on established clinical "
            "knowledge to answer the full question."
        )
    return "\n\n".join([
        "Instructions for Clinical Synthesis:",
        grounding,
        _SYNTHESIS_COVERAGE,
        _SYNTHESIS_LENGTH,
        _SYNTHESIS_STRUCTURE,
    ])


def _evidence_source(context: Dict[str, Any]) -> str:
    """Provenance label for an evidence item (knowledge graph vs literature)."""
    meta = context.get("metadata") or {}
    source = context.get("source", "")
    relation = meta.get("display_relation") or meta.get("relation") or ""
    if source == "nary_fact":
        label = "PrimeKG n-ary fact"
    elif meta.get("doc_type") == "clinical_text" or meta.get("source") == "pmc":
        return "PubMed Central article"
    elif relation or source in ("graph", "graph_hyperedge"):
        label = "PrimeKG knowledge graph"
    else:
        return "clinical literature"
    return f"{label}, relation: {relation}" if relation else label


COT_EXEMPLARS = [
    {
        "question": "A 45-year-old man presents with acute substernal chest pain. ECG shows ST elevation in leads II, III, and aVF. Which immediate medication reduces mortality in this setting?",
        "options": {
            "A": "Aspirin (chewable)",
            "B": "Sublingual nitroglycerin",
            "C": "Intravenous morphine",
            "D": "Oral metoprolol"
        },
        "reasoning": "In acute STEMI, prompt antiplatelet therapy with chewable aspirin significantly reduces acute mortality. Therefore, the correct answer is A.",
        "answer_letter": "A"
    },
    {
        "question": "A 62-year-old woman is prescribed clarithromycin for a respiratory tract infection. She is concurrently taking simvastatin for hypercholesterolemia. Two weeks later, she presents with severe muscle pain and dark-colored urine. Which mechanism best explains her current symptoms?",
        "options": {
            "A": "Induction of cytochrome P450 3A4",
            "B": "Inhibition of cytochrome P450 3A4",
            "C": "Decreased renal clearance of simvastatin",
            "D": "Direct myotoxicity of clarithromycin"
        },
        "reasoning": "Clarithromycin is a potent inhibitor of hepatic CYP3A4. Simvastatin is metabolized by CYP3A4, so inhibition leads to toxicity and rhabdomyolysis. Therefore, the correct answer is B.",
        "answer_letter": "B"
    },
    {
        "question": "A 28-year-old woman presents with persistent chronic diarrhea, abdominal bloating, and weight loss. Duodenal biopsy demonstrates villous atrophy. Which antibody test is most sensitive and specific?",
        "options": {
            "A": "Antinuclear antibody (ANA)",
            "B": "Anti-Saccharomyces cerevisiae antibody (ASCA)",
            "C": "Anti-tissue transglutaminase (anti-tTG) IgA",
            "D": "Anti-neutrophil cytoplasmic antibody (p-ANCA)"
        },
        "reasoning": "Villous atrophy on duodenal biopsy is characteristic of celiac disease. The most sensitive and specific initial screening test is anti-tTG IgA. Therefore, the correct answer is C.",
        "answer_letter": "C"
    },
    {
        "question": "A 10-year-old boy develops urticaria, wheezing, and facial edema within 15 minutes of eating peanuts. Which immune mechanism is primarily responsible?",
        "options": {
            "A": "Cytotoxic antibody-mediated reaction (Type II)",
            "B": "Immune complex deposition (Type III)",
            "C": "Cell-mediated delayed hypersensitivity (Type IV)",
            "D": "IgE-mediated mast cell degranulation (Type I)"
        },
        "reasoning": "Anaphylaxis to peanuts is an immediate Type I hypersensitivity reaction mediated by preformed allergen-specific IgE causing mast cell degranulation. Therefore, the correct answer is D.",
        "answer_letter": "D"
    }
]

COT_SYSTEM_PROMPT = (
    "You are an expert physician answering USMLE clinical questions. "
    "Think step-by-step and state your final answer as: Therefore, the correct answer is [LETTER]."
)

class PromptBuilder:
    """Builds LLM prompts from retrieved contexts."""

    def __init__(self, system_prompt: str = LLM_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def format_context(self, context: dict, index: int = 0) -> str:
        """Format a single context dict as a readable evidence string.

        One line of provenance (source + PrimeKG relation) and the evidence
        text itself. The previous format repeated every triple twice
        ("Verified Biomedical Relation ..." + "Clinical Context ...") and
        labelled unverified retrieval output as "Verified".
        """
        # explicit None checks - a legitimate 0.0 score
        # is not falsy-overridden by a secondary score.
        score = context.get("phi_final")
        if score is None:
            score = context.get("score")
        if score is None:
            score = 0.0

        header = f"[Evidence {index}] ({_evidence_source(context)}; relevance {score:.3f})"
        return f"{header}\n{context_document(context)}"

    def format_contexts_block(self, contexts: list[dict]) -> str:
        """Format a list of contexts as a numbered evidence block."""
        if not contexts:
            return "(No relevant context retrieved.)"
        lines = []
        for i, ctx in enumerate(contexts, 1):
            lines.append(self.format_context(ctx, index=i))
        return "\n\n".join(lines)

    def build_rag_prompt(
        self,
        query: str,
        contexts: list[dict],
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Build the clinical synthesis prompt with retrieved context evidence.

        An empty context list (gate abstention) yields the same no-evidence
        instructions the baseline receives - parametric mode is never worse
        prompted than the no-retrieval arm.
        """
        context_block = self.format_contexts_block(contexts)

        user_content = (
            f"Clinical Query:\n{query}\n\n"
            f"Retrieved Evidence:\n{context_block}\n\n"
            f"{synthesis_instructions('Evidence' if contexts else None)}"
        )

        return [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_mcq_prompt(
        self,
        question: str,
        options: dict,
        contexts: Optional[list[dict]] = None,
        few_shot: bool = False,
        n_exemplars: int = FEW_SHOT_COUNT,  # wire the config constant
    ) -> list[dict]:
        """Build a multiple-choice question prompt for MedQA evaluation.

        Args:
            question: Clinical vignette / question stem.
            options: Dict of answer options, e.g. {"A": "...", "B": "...", ...}
            contexts: Optional retrieved contexts (None for baseline LLM).
            few_shot: If True, prepend balanced CoT exemplars.
            n_exemplars: Number of exemplars to include (max 4).

        Returns:
            Messages list for Ollama.
        """
        options_text = "\n".join(
            f"  {letter}: {text}" for letter, text in sorted(options.items())
        )

        if contexts:
            context_block = self.format_contexts_block(contexts)
            user_content = (
                f"Relevant Context:\n{context_block}\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Based on the provided context and your clinical knowledge, "
                "select the single best answer. If the context does not directly answer the question, "
                "rely on your medical expertise. "
                "Reply with ONLY the letter of the correct answer (A, B, C, or D)."
            )
        else:
            user_content = (
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Select the single best answer. "
                "Reply with ONLY the letter of the correct answer (A, B, C, or D)."
            )

        if few_shot:
            system = COT_SYSTEM_PROMPT
            messages = [{"role": "system", "content": system}]
            for ex in COT_EXEMPLARS[:n_exemplars]:
                opts_text = "\n".join(
                    f"  {k}: {v}" for k, v in sorted(ex["options"].items())
                )
                ex_msg = f"Question:\n{ex['question']}\n\nOptions:\n{opts_text}\n"
                messages.append({"role": "user", "content": ex_msg})
                messages.append({"role": "assistant", "content": ex["reasoning"]})
            messages.append({"role": "user", "content": user_content})
            return messages

        system = (
            "You are a medical expert answering USMLE-style multiple choice questions. "
            "Respond with only a single letter: A, B, C, or D."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    # Hallucination Detection Prompt (MedHallu)

    def build_hallucination_prompt(
        self, question: str, answer: str, knowledge: str
    ) -> list[dict]:
        """
        Build prompt for hallucination detection (MedHallu benchmark).

        Args:
            question: Medical question.
            answer: Candidate answer to evaluate.
            knowledge: Ground-truth supporting knowledge.

        Returns:
            Messages list for Ollama.
        """
        user_content = (
            f"Reference Knowledge:\n{knowledge}\n\n"
            f"Question:\n{question}\n\n"
            f"Candidate Answer:\n{answer}\n\n"
            "Determine if the candidate answer contains hallucinations "
            "(factual errors, unsupported claims, or contradictions with "
            "the reference knowledge). "
            "Reply with ONLY one word: HALLUCINATED or FAITHFUL."
        )

        return [
            {
                "role": "system",
                "content": "You are a medical fact-checker. Classify answers as HALLUCINATED or FAITHFUL.",
            },
            {"role": "user", "content": user_content},
        ]
