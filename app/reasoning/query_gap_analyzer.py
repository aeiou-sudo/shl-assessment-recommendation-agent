"""Generate clarification questions for weak queries before ambiguity resolution."""

from __future__ import annotations

from app.llm.groq_client import GroqReasoningClient
from app.models.catalog import FilteredCandidate
from app.reasoning.query_strength_engine import QueryStrengthReport
from app.state.conversation_state import ConversationState


class QueryGapAnalyzer:
    def __init__(self, llm: GroqReasoningClient | None = None) -> None:
        self.llm = llm or GroqReasoningClient()

    def question(
        self,
        state: ConversationState,
        filtered: list[FilteredCandidate],
        strength: QueryStrengthReport,
    ) -> str:
        fallback = self._fallback_question(strength)
        payload = {
            "state_terms": state.searchable_terms(),
            "shared_terms": strength.shared_terms,
            "candidates": [
                {
                    "name": item.candidate.entry.name,
                    "description": item.candidate.entry.description,
                    "keys": list(item.candidate.entry.keys),
                }
                for item in filtered[:5]
                if item.keep
            ],
        }
        system = (
            "Generate one concise conversational clarification question to strengthen "
            "a weak HR assessment search query. Ask about recurring semantic concepts "
            "in the candidates. Do not offer a rigid menu. Return JSON: {question}."
        )
        parsed = self.llm.json_call(system, payload, {"question": fallback})
        return str(parsed.get("question", fallback)).strip() or fallback

    def _fallback_question(self, strength: QueryStrengthReport) -> str:
        terms = [term for term in strength.shared_terms[:3] if term]
        if terms:
            return f"Should the assessment focus on {', '.join(terms)}, or a different specialization?"
        return "What core role, domain, or skill should the assessment focus on?"
