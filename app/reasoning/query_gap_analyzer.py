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
        fallback = self._fallback_question(state, strength)
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
            "Generate one concise recruiter-facing clarification question. Sound like "
            "an experienced assessment consultant narrowing hiring intent, not a search "
            "system. Ask about role purpose, seniority, domain, specialization, work "
            "context, or assessment use case. Do not mention retrieval, confidence, "
            "keywords, semantic search, candidates, or database mechanics. Return JSON: "
            "{question}."
        )
        parsed = self.llm.json_call(system, payload, {"question": fallback})
        return str(parsed.get("question", fallback)).strip() or fallback

    def _fallback_question(
        self, state: ConversationState, strength: QueryStrengthReport
    ) -> str:
        state_terms = " ".join(state.searchable_terms()).casefold()
        retrieved_terms = " ".join(strength.shared_terms).casefold()
        terms = f"{state_terms} {retrieved_terms}"
        if any(term in state_terms for term in ("leadership", "executive", "director", "manager")):
            return (
                "Is this intended for executive selection, promotion evaluation, "
                "or leadership development?"
            )
        if any(
            term in state_terms
            for term in ("net", "java", "python", "developer", "engineer", "backend", "software")
        ):
            return (
                "Would success in this role depend more on hands-on implementation, "
                "systems design, or operational ownership?"
            )
        if any(term in state_terms for term in ("data", "analytics", "model", "machine", "ml")):
            return (
                "Should the candidate be assessed more for data analysis, predictive "
                "modeling, or production ML delivery?"
            )
        if any(term in terms for term in ("net", "java", "python", "developer", "engineer", "backend")):
            return (
                "Would success in this role depend more on hands-on implementation, "
                "systems design, or operational ownership?"
            )
        if any(term in terms for term in ("sales", "customer", "support", "service")):
            return (
                "Is the role more focused on customer-facing communication, sales "
                "execution, or operational service delivery?"
            )
        return "What core role, domain, or skill should the assessment focus on?"
