"""Build the positive semantic search query from state."""

from __future__ import annotations

from app.state.conversation_state import ConversationState


class QuerySynthesizer:
    """Negative constraints are deliberately excluded from this query."""

    def synthesize(self, state: ConversationState) -> str:
        terms = state.searchable_terms()
        return "\n".join(terms)
