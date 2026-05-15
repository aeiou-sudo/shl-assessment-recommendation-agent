"""Public service facade for the state-driven convergence agent."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.models.catalog import CatalogueEntry
from app.rag.faiss_store import FaissCatalogueStore
from app.reasoning.convergence_engine import (
    ConvergenceEngine,
    ConvergenceStepResult,
    StepStatus,
)
from app.retrieval.semantic_search import SemanticSearch
from app.state.conversation_state import ConversationState


@dataclass
class AgentResponse:
    status: StepStatus
    message: str
    assessment_plan: dict[str, object] | None
    closest_matches: list[dict[str, object]]
    state: ConversationState


class AssessmentRecommendationAgent:
    def __init__(self, convergence: ConvergenceEngine) -> None:
        self.convergence = convergence

    @classmethod
    def from_catalogue(cls) -> "AssessmentRecommendationAgent":
        if (settings.index_dir / "catalogue.faiss").exists():
            store = FaissCatalogueStore.load(settings.index_dir)
        else:
            store = FaissCatalogueStore.from_catalogue()
        return cls(ConvergenceEngine(SemanticSearch(store)))

    def handle(self, query: str, state: ConversationState | None = None) -> AgentResponse:
        state = state or ConversationState()
        result = self.convergence.run_turn(query, state)
        plan = None
        if result.status == StepStatus.CONVERGED:
            top = _top_kept(result)
            if top:
                plan = build_assessment_plan(top, state)
                result = ConvergenceStepResult(
                    status=result.status,
                    message=f"Recommended assessment plan for {top.name}.",
                    filtered_candidates=result.filtered_candidates,
                    confidence=result.confidence,
                    strength=result.strength,
                    ambiguity=result.ambiguity,
                    trajectory=result.trajectory,
                )
        return AgentResponse(
            status=result.status,
            message=result.message,
            assessment_plan=plan,
            closest_matches=_matches(result),
            state=state,
        )


def build_assessment_plan(
    entry: CatalogueEntry,
    state: ConversationState,
) -> dict[str, object]:
    """Generate final output from the matched catalogue entry only."""
    return {
        "matched_catalogue_entry": {
            "entity_id": entry.entity_id,
            "name": entry.name,
            "link": entry.link,
            "description": entry.description,
        },
        "assessment_scope": {
            "catalogue_categories": list(entry.keys),
            "job_levels": list(entry.job_levels),
            "duration": entry.duration,
            "remote": entry.remote,
            "adaptive": entry.adaptive,
            "languages": list(entry.languages),
        },
        "recruiter_clarifications_preserved": {
            "domains": state.domains,
            "skills": state.skills,
            "specializations": state.specializations,
            "linked_intents": [intent.label for intent in state.linked_intents],
            "negative_constraints": state.negative_constraints,
            "clarification_answers": state.clarification_answers,
        },
        "recommended_use": (
            "Use this catalogue assessment as the primary assessment component. "
            "Pair it with recruiter-specific interview questions only where the "
            "catalogue description does not claim coverage."
        ),
    }


def _top_kept(result: ConvergenceStepResult) -> CatalogueEntry | None:
    for item in result.filtered_candidates:
        if item.keep:
            return item.candidate.entry
    return None


def _matches(result: ConvergenceStepResult) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for item in result.filtered_candidates:
        if not item.keep:
            continue
        entry = item.candidate.entry
        matches.append(
            {
                "entity_id": entry.entity_id,
                "name": entry.name,
                "score": round(item.candidate.score, 4),
                "description": entry.description,
                "link": entry.link,
                "filter_reason": item.reason,
            }
        )
    return matches[:5]
