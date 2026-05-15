"""Structured state for the retrieval convergence conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class QueryType(StrEnum):
    OUT_OF_CONTEXT = "OUT_OF_CONTEXT"
    NEW_QUERY = "NEW_QUERY"
    REFINEMENT_QUERY = "REFINEMENT_QUERY"
    INTENT_SHIFT = "INTENT_SHIFT"
    CLARIFICATION_RESPONSE = "CLARIFICATION_RESPONSE"


@dataclass
class IntentState:
    label: str
    domains: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    specializations: list[str] = field(default_factory=list)
    raw_phrases: list[str] = field(default_factory=list)
    active: bool = True


@dataclass
class ConfidenceSnapshot:
    top_score: float
    spread: float
    entropy: float
    reason: str


@dataclass
class ConversationState:
    primary_intents: list[IntentState] = field(default_factory=list)
    linked_intents: list[IntentState] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    specializations: list[str] = field(default_factory=list)
    rejected_intents: list[str] = field(default_factory=list)
    ambiguity_history: list[str] = field(default_factory=list)
    clarification_answers: list[str] = field(default_factory=list)
    confidence_history: list[ConfidenceSnapshot] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    last_query_type: QueryType | None = None
    last_user_query: str = ""
    turns: int = 0

    def searchable_terms(self) -> list[str]:
        """Return positive intent terms only, excluding negative constraints."""
        terms: list[str] = []
        for intent in [*self.primary_intents, *self.linked_intents]:
            if intent.active:
                terms.extend([intent.label, *intent.domains, *intent.skills])
                terms.extend(intent.specializations)
                terms.extend(intent.raw_phrases)
        terms.extend(self.domains)
        terms.extend(self.skills)
        terms.extend(self.specializations)
        return _dedupe([term for term in terms if term.strip()])


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result
