"""
Abstract base class for all LLM provider implementations — Python 3.10.

The rest of the system depends only on this interface, never on Ollama
or any other concrete library.  Swapping providers requires only a new
subclass; no other file changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Minimal contract that every LLM backend must satisfy.

    Design note
    -----------
    The interface is intentionally thin (one method) so wrapping new
    backends stays trivial.  Provider-specific concerns — batching,
    streaming, tool-use — live inside the concrete class, not here.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to the language model and return generated text.

        Parameters
        ----------
        prompt:
            The fully-assembled prompt string (system + context + query).

        Returns
        -------
        str
            The model's response, leading/trailing whitespace stripped.

        Raises
        ------
        RuntimeError
            If the underlying provider returns an error or unexpected
            payload that cannot be recovered from.
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks for v2 streaming / tool-use extensions
    # ------------------------------------------------------------------

    def supports_streaming(self) -> bool:
        """Return ``True`` if this provider supports token-by-token streaming."""
        return False

    def model_name(self) -> str:
        """Return a human-readable identifier for the active model."""
        return self.__class__.__name__
