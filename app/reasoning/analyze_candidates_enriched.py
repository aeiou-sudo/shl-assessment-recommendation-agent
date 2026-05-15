"""Ambiguity analysis across high-ranking catalogue candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm.groq_client import GroqReasoningClient
from app.models.catalog import FilteredCandidate
from app.state.conversation_state import ConversationState


@dataclass(frozen=True)
class AmbiguityReport:
    ambiguous: bool
    differentiators: list[str]
    ambiguity_terms: list[str]
    question: str


class CandidateAmbiguityAnalyzer:
    def __init__(self, llm: GroqReasoningClient | None = None) -> None:
        self.llm = llm or GroqReasoningClient()

    def analyze(
        self,
        state: ConversationState,
        filtered: list[FilteredCandidate],
    ) -> AmbiguityReport:
        kept = [item.candidate for item in filtered if item.keep]
        fallback = self._fallback(kept)
        payload = {
            "state_terms": state.searchable_terms(),
            "negative_constraints": state.negative_constraints,
            "candidates": [
                {
                    "name": item.entry.name,
                    "description": item.entry.description,
                    "keys": list(item.entry.keys),
                    "score": item.score,
                }
                for item in kept[:5]
            ],
        }
        system = (
            "Compare top SHL catalogue candidates and identify ambiguity. Return JSON "
            "with ambiguous, differentiators, ambiguity_terms, question. Ask one focused "
            "domain- or specialization-level clarification question. Do not invent facts."
        )
        parsed = self.llm.json_call(system, payload, fallback.__dict__)
        return AmbiguityReport(
            ambiguous=bool(parsed.get("ambiguous", fallback.ambiguous)),
            differentiators=_list(parsed.get("differentiators", fallback.differentiators)),
            ambiguity_terms=_list(parsed.get("ambiguity_terms", fallback.ambiguity_terms)),
            question=str(parsed.get("question", fallback.question)).strip()
            or fallback.question,
        )

    def _fallback(self, kept: list[object]) -> AmbiguityReport:
        if len(kept) < 2:
            return AmbiguityReport(False, [], [], "")
        top_terms = [_important_terms(item.entry.name + " " + item.entry.description) for item in kept[:3]]
        differentiators = sorted(set().union(*top_terms))[:8]
        names = [item.entry.name for item in kept[:2]]
        question = f"Are you closer to {names[0]} or {names[1]}?"
        return AmbiguityReport(True, differentiators, differentiators[:4], question)


def _important_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{2,}", text.casefold())
        if token not in {"the", "and", "for", "with", "test", "knowledge", "measures"}
    }


def _list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
