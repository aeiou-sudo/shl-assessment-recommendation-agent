"""Runtime configuration for the assessment convergence agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    catalogue_path: Path = ROOT_DIR / "data" / "shl_product_catalog.json"
    index_dir: Path = ROOT_DIR / "data" / "indexes"
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    llm_temperature: float = 0.000001
    default_top_k: int = 8
    convergence_threshold: float = 0.72
    ambiguity_margin: float = 0.08
    weak_query_threshold: float = 0.45


settings = Settings()
