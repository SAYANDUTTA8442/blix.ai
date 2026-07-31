"""
HuggingFace Transformers-backed LLMProvider — v0.2.

Uses a local causal language model loaded via ``transformers`` + ``torch``
for fully offline inference.  No Ollama required.

Model is lazy-loaded on first ``generate()`` call so startup is instant.

Supported architectures
-----------------------
Any ``AutoModelForCausalLM``-compatible checkpoint, e.g.:
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0   (recommended, ~2GB)
  - microsoft/phi-2
  - HuggingFaceTB/SmolLM2-360M-Instruct  (lightweight, ~720MB)
  - meta-llama/Llama-3.2-1B-Instruct     (needs HF_TOKEN)
"""

from __future__ import annotations

import os
from typing import Optional, Any

from llm.base import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)


class TransformersProvider(LLMProvider):
    """
    Runs a causal LM locally via HuggingFace ``transformers``.

    Parameters
    ----------
    model_name:
        HuggingFace Hub model id or local path.
    temperature:
        Sampling temperature (0.0 → greedy, >0 → sampling).
    max_new_tokens:
        Maximum tokens to generate per call.
    device:
        ``"cpu"``, ``"cuda"``, or ``"auto"`` (default).
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._device = device

        # Lazy-loaded on first generate() call
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._pipe: Optional[Any] = None

        log.info("TransformersProvider configured (model=%s)", model_name)

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load tokenizer + model into memory (called once on first use)."""
        if self._pipe is not None:
            return

        log.info("Loading model %r … (this may take a minute on first run)", self._model_name)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            hf_token = os.getenv("HF_TOKEN") or None

            tok = AutoTokenizer.from_pretrained(
                self._model_name,
                token=hf_token,
                trust_remote_code=True,
            )
            # Resolve device
            if self._device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = self._device

            model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                token=hf_token,
                torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
                device_map=dev,
                trust_remote_code=True,
            )
            model.eval()

            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                device_map=dev,
            )
            self._tokenizer = tok
            log.info("Model loaded on device=%s", dev)

        except ImportError as exc:
            raise RuntimeError(
                "torch / transformers not installed.\n"
                "Run: pip install torch transformers"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model {self._model_name!r}: {exc}\n"
                "Check HF_TOKEN in .env for gated models, or choose a public model."
            ) from exc

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Run local inference on *prompt* and return the generated text.

        The prompt is wrapped in the chat template if the tokenizer
        supports it; otherwise sent as plain text.
        """
        self._load()

        try:
            # Use apply_chat_template if available (instruction-tuned models)
            if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = prompt

            do_sample = self._temperature > 0.0
            outputs = self._pipe(
                formatted,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature if do_sample else None,
                do_sample=do_sample,
                return_full_text=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            text: str = outputs[0]["generated_text"]
            return text.strip()

        except Exception as exc:
            log.error("TransformersProvider.generate failed: %s", exc)
            raise RuntimeError(f"TransformersProvider.generate failed: {exc}") from exc

    # ------------------------------------------------------------------
    # LLMProvider overrides
    # ------------------------------------------------------------------

    def model_name(self) -> str:
        return self._model_name

    def supports_streaming(self) -> bool:
        return False  # v0.3: add TextIteratorStreamer

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` after the first ``generate()`` call."""
        return self._pipe is not None
