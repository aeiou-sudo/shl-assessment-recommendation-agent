"""Build and persist the FAISS catalogue index."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.faiss_store import FaissCatalogueStore


def main() -> None:
    store = FaissCatalogueStore.from_catalogue()
    store.save()
    print(f"Indexed {len(store.documents)} semantic documents from {len(store.entries)} entries.")


if __name__ == "__main__":
    main()
