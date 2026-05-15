"""Thin deterministic retrieval facade."""

from __future__ import annotations

from app.core.config import settings
from app.models.catalog import RetrievalCandidate
from app.rag.faiss_store import FaissCatalogueStore
from app.retrieval.query_synthesizer import QuerySynthesizer
from app.state.conversation_state import ConversationState


class SemanticSearch:
    def __init__(
        self,
        store: FaissCatalogueStore,
        synthesizer: QuerySynthesizer | None = None,
    ) -> None:
        self.store = store
        self.synthesizer = synthesizer or QuerySynthesizer()

    def retrieve(
        self, state: ConversationState, top_k: int = settings.default_top_k
    ) -> list[RetrievalCandidate]:
        query = self.synthesizer.synthesize(state)
        return self.store.search(query, top_k=top_k)
