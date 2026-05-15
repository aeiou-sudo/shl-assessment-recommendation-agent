"""Deterministic embedding store backed by FAISS."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from app.core.config import settings
from app.models.catalog import CatalogueEntry, RetrievalCandidate, SemanticDocument
from app.rag.catalogue_processor import CatalogueProcessor


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = settings.embedding_model) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = int(self.model.get_embedding_dimension())
        else:
            self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")


class HashingEmbedder:
    """Offline deterministic fallback used when no embedding model is available."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}", text.casefold()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class FaissCatalogueStore:
    def __init__(
        self,
        entries: list[CatalogueEntry],
        documents: list[SemanticDocument],
        embedder: Embedder | None = None,
    ) -> None:
        self.entries = {entry.entity_id: entry for entry in entries}
        self.documents = documents
        self.embedder = embedder or build_embedder()
        self.index = faiss.IndexFlatIP(self.embedder.dimension)

    @classmethod
    def from_catalogue(
        cls, catalogue_path: Path = settings.catalogue_path, embedder: Embedder | None = None
    ) -> "FaissCatalogueStore":
        processor = CatalogueProcessor()
        entries = processor.load_entries(catalogue_path)
        documents = processor.build_documents(entries)
        store = cls(entries, documents, embedder)
        store.build()
        return store

    def build(self) -> None:
        embeddings = self.embedder.encode([doc.text for doc in self.documents])
        self.index.reset()
        self.index.add(embeddings)

    def save(self, index_dir: Path = settings.index_dir) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "catalogue.faiss"))
        (index_dir / "documents.pkl").write_bytes(pickle.dumps(self.documents))
        entries_payload = [entry.raw for entry in self.entries.values()]
        (index_dir / "entries.json").write_text(
            json.dumps(entries_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, index_dir: Path = settings.index_dir, embedder: Embedder | None = None
    ) -> "FaissCatalogueStore":
        processor = CatalogueProcessor()
        entries = processor.load_entries(index_dir / "entries.json")
        documents = pickle.loads((index_dir / "documents.pkl").read_bytes())
        store = cls(entries, documents, embedder)
        store.index = faiss.read_index(str(index_dir / "catalogue.faiss"))
        return store

    def search(self, query: str, top_k: int = settings.default_top_k) -> list[RetrievalCandidate]:
        if not query.strip():
            return []
        query_vector = self.embedder.encode([query])
        scores, indexes = self.index.search(query_vector, min(len(self.documents), top_k * 4))
        by_entity: dict[str, RetrievalCandidate] = {}
        for score, idx in zip(scores[0], indexes[0]):
            if idx < 0:
                continue
            document = self.documents[int(idx)]
            entry = self.entries[document.entity_id]
            adjusted_score = min(1.0, float(score) + _lexical_boost(query, entry))
            current = by_entity.get(entry.entity_id)
            candidate = RetrievalCandidate(
                entry=entry,
                document=document,
                score=adjusted_score,
                rank=0,
            )
            if current is None or candidate.score > current.score:
                by_entity[entry.entity_id] = candidate
        ranked = sorted(by_entity.values(), key=lambda item: item.score, reverse=True)[:top_k]
        return [
            RetrievalCandidate(item.entry, item.document, item.score, rank)
            for rank, item in enumerate(ranked, start=1)
        ]


def build_embedder() -> Embedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashingEmbedder()


def _lexical_boost(query: str, entry: CatalogueEntry) -> float:
    query_norm = _norm(query)
    name_norm = _norm(entry.name)
    if name_norm and name_norm in query_norm:
        return 0.25
    name_tokens = {token for token in name_norm.split() if len(token) > 1}
    query_tokens = set(query_norm.split())
    if name_tokens and name_tokens.issubset(query_tokens):
        return 0.16
    overlap = len(name_tokens & query_tokens)
    if overlap >= 2:
        return 0.06
    return 0.0


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9+#]+", text.casefold()))
