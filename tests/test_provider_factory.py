"""Tests for llm/provider_factory.py"""

from __future__ import annotations

import pytest

from config.settings import LLMSettings
from llm.provider_factory import build_provider
from llm.ollama_provider import OllamaProvider
from llm.transformers_provider import TransformersProvider


class TestBuildProvider:
    def test_builds_transformers(self) -> None:
        cfg = LLMSettings(provider="transformers", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        provider = build_provider(cfg)
        assert isinstance(provider, TransformersProvider)

    def test_builds_ollama(self) -> None:
        cfg = LLMSettings(provider="ollama", ollama_model="llama3.2")
        provider = build_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_unknown_provider_raises(self) -> None:
        cfg = LLMSettings(provider="gpt-99")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_provider(cfg)

    def test_case_insensitive(self) -> None:
        cfg = LLMSettings(provider="Transformers", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        provider = build_provider(cfg)
        assert isinstance(provider, TransformersProvider)
