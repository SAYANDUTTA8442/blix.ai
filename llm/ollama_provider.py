"""
Ollama-backed LLMProvider implementation — Python 3.10 compatible.

Uses ``ollama.chat()`` under the hood.  All Ollama-specific concerns
(message format, option keys, error shapes) are confined to this module.
"""

from __future__ import annotations

from typing import Any

from llm.base import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """
    Sends prompts to a locally-running Ollama instance.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.2"``, ``"mistral"``,
        ``"gemma3"``.
    temperature:
        Sampling temperature forwarded as an Ollama option.
    max_tokens:
        Maximum tokens to generate (``num_predict`` in Ollama).
    """

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        log.info("OllamaProvider initialised (model=%s)", model)

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to Ollama and return the assistant's reply.

        The full context block assembled by ``PromptBuilder`` is sent as
        a single ``user`` message; Ollama handles the role separation
        internally.

        Raises
        ------
        RuntimeError
            On import failure (ollama not installed) or API errors.
        """
        try:
            import ollama  # lazy import — keeps the rest of the app importable without ollama
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' Python package is not installed.\n"
                "Run: pip install ollama"
            ) from exc

        try:
            response: Any = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            )
            text: str = response["message"]["content"]
            return text.strip()
        except KeyError as exc:
            raise RuntimeError(
                f"Unexpected Ollama response shape — missing key {exc}.\n"
                "Check that your Ollama version is up to date."
            ) from exc
        except Exception as exc:
            log.error("Ollama generation failed: %s", exc)
            raise RuntimeError(
                f"OllamaProvider.generate failed: {exc}\n"
                "Is Ollama running?  Try: ollama serve"
            ) from exc

    # ------------------------------------------------------------------
    # LLMProvider overrides
    # ------------------------------------------------------------------

    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Active Ollama model tag."""
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens
