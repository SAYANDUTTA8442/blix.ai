"""
LLMProviderFactory — creates the correct LLMProvider from settings.  v0.2

Centralises provider selection so ``app.py`` never imports Ollama or
Transformers directly.

Python 3.10 compatible.
"""

from __future__ import annotations

from config.settings import LLMSettings
from llm.base import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)


def build_provider(cfg: LLMSettings) -> LLMProvider:
    """
    Instantiate and return the configured ``LLMProvider``.

    Parameters
    ----------
    cfg:
        ``LLMSettings`` from the loaded ``AppSettings``.

    Returns
    -------
    LLMProvider
        A ready-to-use provider instance (model is NOT yet loaded into
        memory for the Transformers backend — that happens on first call).

    Raises
    ------
    ValueError
        If ``cfg.provider`` is not a recognised value.
    """
    provider = cfg.provider.lower().strip()
    log.info("Building LLM provider: %r", provider)

    if provider == "transformers":
        from llm.transformers_provider import TransformersProvider
        return TransformersProvider(
            model_name=cfg.model,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
        )

    if provider == "ollama":
        from llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            model=cfg.ollama_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
        )

    raise ValueError(
        f"Unknown LLM provider {cfg.provider!r}. "
        "Valid options: 'transformers', 'ollama'."
    )
