import logging
import re
from typing import Optional

import requests

from src.config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

# Ollama API defaults
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

# System prompt for the standalone baseline - matches HCAD-RAG structure
BASELINE_SYSTEM_PROMPT = (
    "You are a biomedical research assistant specializing in clinical medicine. "
    "Use your expert clinical domain knowledge to deliver comprehensive, accurate, "
    "and actionable answers. Be precise, thorough, and clinically rigorous."
)

def _extract_answer_letter(response: str) -> str:
    """Robustly extract a single answer letter (A/B/C/D) from varied LLM outputs.

    Handles common LLM response patterns including:
      - "The answer is B"
      - "B) ..."
      - "Option B"
      - "**B**"
      - "Answer: B"
      - Single letter response "B"
      - "B."
      - "(B)"

    Args:
        response: Raw LLM text response.

    Returns:
        Uppercase answer letter (A, B, C, or D), or "X" if extraction fails.
    """
    if not response or not response.strip():
        logger.warning("Empty response received for answer extraction")
        return "X"

    text = response.strip()

    # Strip DeepSeek-R1 <think>...</think> reasoning block if present
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Pattern priority order (most specific → least specific):

    # 1. "The answer is [X]" / "the correct answer is [X]"
    match = re.search(
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\-]?\s*\(?([A-Da-d])\)?",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # 2. "Answer: X" or "Answer - X"
    match = re.search(r"answer\s*[:\-]\s*\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. "Option X" or "option (X)"
    match = re.search(r"option\s*\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 4. Markdown bold "**X**"
    match = re.search(r"\*\*([A-Da-d])\*\*", text)
    if match:
        return match.group(1).upper()

    # 5. Letter followed by paren or period at start of line: "B)" or "B."
    match = re.search(r"(?:^|\n)\s*\(?([A-Da-d])\)?[\.\)\:]", text)
    if match:
        return match.group(1).upper()

    # 6. Standalone letter on its own line
    match = re.search(r"(?:^|\n)\s*([A-Da-d])\s*(?:$|\n)", text)
    if match:
        return match.group(1).upper()

    # 7. Letter at end of text or after colon/dash (last resort, avoids article "A")
    match = re.search(r"(?:[:\-]\s*|(?:^|\n)\s*)([A-Da-d])\s*$", text)
    if match:
        return match.group(1).upper()

    logger.warning("Could not extract answer letter from response: %.100s...", text)
    return "X"

extract_answer_letter = _extract_answer_letter

class BaselineLLM:
    """Standalone LLM baseline with no retrieval augmentation.

    This baseline directly queries an Ollama-hosted LLM to establish a
    performance floor for medical QA. It demonstrates what a foundation
    model can achieve purely from its parametric knowledge without any
    external retrieval context.

    Attributes:
        model: Name of the Ollama model (from config.LLM_MODEL).
        temperature: Sampling temperature for generation.
        max_tokens: Maximum number of tokens to generate.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        """Initialize the standalone LLM baseline.

        Args:
            model: Ollama model identifier (default: config.LLM_MODEL).
            temperature: Sampling temperature (default: config.LLM_TEMPERATURE).
            max_tokens: Max generation tokens (default: config.LLM_MAX_TOKENS).
        """
        self.model = model or LLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(
            "BaselineLLM initialized - model=%s, temp=%.2f, max_tokens=%d",
            self.model, self.temperature, self.max_tokens,
        )

    def _call_ollama(self, prompt: str, system: str = BASELINE_SYSTEM_PROMPT) -> str:
        """Send a prompt to the Ollama API and return the generated text.

        Args:
            prompt: User prompt to send.
            system: System-level instruction string.

        Returns:
            Generated text response.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
            RuntimeError: If the API returns an error status.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            resp = requests.post(
                OLLAMA_GENERATE_ENDPOINT,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            logger.error(
                "Cannot connect to Ollama at %s. Is `ollama serve` running?",
                OLLAMA_BASE_URL,
            )
            raise ConnectionError(
                f"Ollama server unreachable at {OLLAMA_BASE_URL}. "
                "Start it with: ollama serve"
            ) from exc
        except requests.HTTPError as exc:
            logger.error("Ollama API error: %s", exc)
            raise RuntimeError(f"Ollama API error: {exc}") from exc

        data = resp.json()
        return (data.get("response") or "").strip()

    def generate(self, query: str) -> str:
        """Generate a free-text answer to a medical query (no retrieval)."""
        from src.core.llm_client import UnifiedLLMClient
        from src.core.prompt_builder import synthesis_instructions

        # Same synthesis instructions as both RAG arms (no-evidence variant).
        user_content = f"Clinical Query:\n{query}\n\n{synthesis_instructions(None)}"
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            client = UnifiedLLMClient(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
            return client.chat(messages)
        except Exception as exc:
            logger.error("BaselineLLM generate failed: %s", exc)
            return f"[Error: {exc}]"
