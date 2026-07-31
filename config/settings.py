"""
Blix v0.2 configuration settings — Python 3.10 compatible.

All settings are read from environment variables (via .env) first,
then fall back to blix.yaml, then to coded defaults.

Priority: .env  >  blix.yaml  >  code defaults

Usage
-----
    from config.settings import settings
    print(settings.embed_model)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root (silently ok if missing)
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT_DIR / ".env", override=False)

ROOT_DIR: Path = _ROOT_DIR
MEMORY_DIR: Path = ROOT_DIR / "memory"
CONFIG_FILE: Path = ROOT_DIR / "config" / "blix.yaml"


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class LLMSettings(BaseModel):
    """Settings for the chat LLM provider."""

    provider: str = Field(default="transformers", description="'transformers' or 'ollama'")
    model: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="HuggingFace model id or Ollama tag.",
    )
    ollama_model: str = Field(default="llama3.2", description="Ollama model tag (ollama provider only).")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=512, gt=0)


class EmbedSettings(BaseModel):
    """Settings for the semantic embedding retriever."""

    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name.",
    )
    threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Cosine similarity cutoff.")
    top_k: int = Field(default=5, gt=0, description="Max results from semantic search.")
    embeddings_file: Path = MEMORY_DIR / "embeddings.npy"
    embedding_ids_file: Path = MEMORY_DIR / "embedding_ids.json"


class MemorySettings(BaseModel):
    """Controls for the memory subsystem."""

    conversations_file: Path = MEMORY_DIR / "conversations.json"
    profile_file: Path = MEMORY_DIR / "profile.json"
    learning_state_file: Path = MEMORY_DIR / "learning_state.json"

    # Legacy retriever knobs (still used as fallback)
    recent_k: int = Field(default=5, gt=0)
    fuzzy_top_k: int = Field(default=3, gt=0)
    fuzzy_threshold: float = Field(default=60.0, ge=0.0, le=100.0)
    keyword_top_k: int = Field(default=3, gt=0)

    # Auto memory extraction
    auto_extract: bool = Field(default=True, description="Run CoT extractor after each turn.")


class AppSettings(BaseModel):
    """Top-level application settings."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embed: EmbedSettings = Field(default_factory=EmbedSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    debug: bool = False


# ---------------------------------------------------------------------------
# Loader — env vars override yaml
# ---------------------------------------------------------------------------


def load_settings(path: Optional[Path] = None) -> AppSettings:
    """
    Build ``AppSettings`` by layering sources:

    1.  YAML file (``config/blix.yaml``) for structured overrides.
    2.  Environment variables (read from ``.env`` + shell) for secrets
        and deployment-specific values.
    3.  Pydantic defaults for anything not specified.
    """
    target = path or CONFIG_FILE
    raw: dict = {}
    if target.exists():
        with target.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    # Env var overrides (dotenv already loaded above)
    env_llm: dict = {}
    if os.getenv("BLIX_LLM_PROVIDER"):
        env_llm["provider"] = os.environ["BLIX_LLM_PROVIDER"]
    if os.getenv("BLIX_LLM_MODEL"):
        env_llm["model"] = os.environ["BLIX_LLM_MODEL"]
    if os.getenv("BLIX_OLLAMA_MODEL"):
        env_llm["ollama_model"] = os.environ["BLIX_OLLAMA_MODEL"]
    if os.getenv("BLIX_TEMPERATURE"):
        env_llm["temperature"] = float(os.environ["BLIX_TEMPERATURE"])
    if os.getenv("BLIX_MAX_NEW_TOKENS"):
        env_llm["max_new_tokens"] = int(os.environ["BLIX_MAX_NEW_TOKENS"])
    if env_llm:
        raw.setdefault("llm", {}).update(env_llm)

    env_embed: dict = {}
    if os.getenv("BLIX_EMBED_MODEL"):
        env_embed["model"] = os.environ["BLIX_EMBED_MODEL"]
    if os.getenv("BLIX_SEMANTIC_THRESHOLD"):
        env_embed["threshold"] = float(os.environ["BLIX_SEMANTIC_THRESHOLD"])
    if os.getenv("BLIX_SEMANTIC_TOP_K"):
        env_embed["top_k"] = int(os.environ["BLIX_SEMANTIC_TOP_K"])
    if env_embed:
        raw.setdefault("embed", {}).update(env_embed)

    if os.getenv("BLIX_AUTO_EXTRACT"):
        val = os.environ["BLIX_AUTO_EXTRACT"].lower()
        raw.setdefault("memory", {})["auto_extract"] = val not in ("0", "false", "no")

    return AppSettings.model_validate(raw)


settings: AppSettings = load_settings()


# ---------------------------------------------------------------------------
# v0.3 settings additions
# ---------------------------------------------------------------------------


class HierarchySettings(BaseModel):
    """Settings for the memory hierarchy manager."""

    hierarchy_dir: Path = MEMORY_DIR / "hierarchy"
    session_idle_gap_minutes: int = Field(
        default=30, description="Gap with no messages that starts a new session."
    )
    auto_daily_rollup: bool = True
    auto_weekly_rollup: bool = True


class GraphSettings(BaseModel):
    """Settings for the memory graph."""

    graph_file: Path = MEMORY_DIR / "graph.json"
    enabled: bool = True


class ProjectSettings(BaseModel):
    """Settings for project memory."""

    projects_file: Path = MEMORY_DIR / "projects.json"


class ProfileSettings(BaseModel):
    """Settings for the versioned profile evolver."""

    versioned_profile_file: Path = MEMORY_DIR / "versioned_profile.json"


class BackgroundSettings(BaseModel):
    """Settings for the background processor."""

    enabled: bool = True
    worker_count: int = Field(default=1, gt=0)
    max_queue_size: int = Field(default=100, gt=0)


class ScoringSettings(BaseModel):
    """Configurable weights for the memory scorer."""

    relevance_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    importance_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    frequency_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    recency_half_life_days: float = Field(default=30.0, gt=0.0)
