"""Transform semi-structured catalogue JSON into semantic documents."""

from __future__ import annotations

import json
from pathlib import Path

from app.models.catalog import CatalogueEntry, SemanticDocument


class CatalogueProcessor:
    """Build searchable documents while preserving source entry references."""

    def load_entries(self, path: Path) -> list[CatalogueEntry]:
        payload = json.JSONDecoder(strict=False).decode(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Catalogue JSON must contain a list of entries.")
        return [CatalogueEntry.from_json(item) for item in payload if isinstance(item, dict)]

    def build_documents(self, entries: list[CatalogueEntry]) -> list[SemanticDocument]:
        documents: list[SemanticDocument] = []
        for entry in entries:
            documents.extend(self._documents_for_entry(entry))
        return documents

    def _documents_for_entry(self, entry: CatalogueEntry) -> list[SemanticDocument]:
        fields = {
            "overview": [
                f"Assessment name: {entry.name}",
                f"Description: {entry.description}",
                f"Assessment categories: {', '.join(entry.keys)}",
            ],
            "role_context": [
                f"Role or skill assessed: {entry.name}",
                f"Job levels: {', '.join(entry.job_levels)}",
                f"Remote: {entry.remote}",
                f"Adaptive: {entry.adaptive}",
                f"Duration: {entry.duration}",
            ],
            "specialization": [
                f"Specialization and domain terms: {entry.name}",
                f"Keywords: {', '.join(entry.keys)}",
                f"Detailed assessment content: {entry.description}",
            ],
        }
        docs: list[SemanticDocument] = []
        for section, parts in fields.items():
            text = "\n".join(part for part in parts if part and not part.endswith(": "))
            docs.append(
                SemanticDocument(
                    doc_id=f"{entry.entity_id}:{section}",
                    entity_id=entry.entity_id,
                    section=section,
                    text=text,
                    metadata={
                        "name": entry.name,
                        "link": entry.link,
                        "keys": list(entry.keys),
                        "job_levels": list(entry.job_levels),
                    },
                )
            )
        return docs
