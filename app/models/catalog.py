"""Catalogue and retrieval data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CatalogueEntry:
    entity_id: str
    name: str
    link: str = ""
    description: str = ""
    keys: tuple[str, ...] = ()
    job_levels: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    duration: str = ""
    remote: str = ""
    adaptive: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CatalogueEntry":
        return cls(
            entity_id=str(payload.get("entity_id", "")),
            name=str(payload.get("name", "")).strip(),
            link=str(payload.get("link", "")).strip(),
            description=str(payload.get("description", "")).strip(),
            keys=tuple(str(x).strip() for x in payload.get("keys", []) if str(x).strip()),
            job_levels=tuple(
                str(x).strip() for x in payload.get("job_levels", []) if str(x).strip()
            ),
            languages=tuple(
                str(x).strip() for x in payload.get("languages", []) if str(x).strip()
            ),
            duration=str(payload.get("duration", "")).strip(),
            remote=str(payload.get("remote", "")).strip(),
            adaptive=str(payload.get("adaptive", "")).strip(),
            raw=payload,
        )


@dataclass(frozen=True)
class SemanticDocument:
    doc_id: str
    entity_id: str
    section: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalCandidate:
    entry: CatalogueEntry
    document: SemanticDocument
    score: float
    rank: int


@dataclass(frozen=True)
class FilteredCandidate:
    candidate: RetrievalCandidate
    keep: bool
    reason: str
    violates_negative_constraints: bool = False
