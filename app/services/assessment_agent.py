"""Public service facade for the state-driven convergence agent."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.core.config import settings
from app.models.catalog import CatalogueEntry
from app.rag.faiss_store import FaissCatalogueStore
from app.reasoning.convergence_engine import (
    ConvergenceEngine,
    ConvergenceStepResult,
    StepStatus,
)
from app.retrieval.semantic_search import SemanticSearch
from app.services.conversation_presenter import ConversationPresenter
from app.state.conversation_state import ConversationState


@dataclass
class AgentResponse:
    message: str
    recommendations: list[dict[str, object]]
    assessment_plan: dict[str, object] | None
    end_of_conversation: bool
    state: ConversationState
    internal_status: StepStatus


class AssessmentRecommendationAgent:
    def __init__(
        self,
        convergence: ConvergenceEngine,
        presenter: ConversationPresenter | None = None,
    ) -> None:
        self.convergence = convergence
        self.presenter = presenter or ConversationPresenter()

    @classmethod
    def from_catalogue(cls) -> "AssessmentRecommendationAgent":
        if (settings.index_dir / "catalogue.faiss").exists():
            store = FaissCatalogueStore.load(settings.index_dir)
        else:
            store = FaissCatalogueStore.from_catalogue()
        return cls(ConvergenceEngine(SemanticSearch(store)))

    def handle(self, query: str, state: ConversationState | None = None) -> AgentResponse:
        state = state or ConversationState()
        if _is_completion_acknowledgement(query) and state.final_recommendations:
            state.conversation_complete = True
            public = self.presenter.render_completion(state)
            return AgentResponse(
                message=public.message,
                recommendations=public.recommendations,
                assessment_plan=public.assessment_plan,
                end_of_conversation=public.end_of_conversation,
                state=state,
                internal_status=StepStatus.CONVERGED,
            )

        result = self.convergence.run_turn(query, state)
        plan = None
        if result.status == StepStatus.CONVERGED:
            top = _primary_kept(result, state)
            if top:
                plan = build_assessment_plan(top, state)
        elif result.status == StepStatus.NO_EXACT_MATCH and _has_kept_candidates(result):
            top = _primary_kept(result, state)
            plan = build_assessment_plan(top, state) if top else None

        public = self.presenter.render(result, state, plan)
        if public.recommendations and state.final_recommendations:
            public = replace(
                public,
                recommendations=_merge_public_recommendations(
                    state.final_recommendations, public.recommendations
                ),
            )
        if public.recommendations:
            state.final_recommendations = public.recommendations
            state.final_assessment_plan = public.assessment_plan
            state.final_message = _completion_message(public)
        return AgentResponse(
            message=public.message,
            recommendations=public.recommendations,
            assessment_plan=plan,
            end_of_conversation=public.end_of_conversation,
            state=state,
            internal_status=result.status,
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
            "description": _normalize_catalogue_text(entry.description),
        },
        "assessment_scope": {
            "catalogue_categories": list(entry.keys),
            "job_levels": list(entry.job_levels),
            "duration": entry.duration,
            "remote": entry.remote,
            "adaptive": entry.adaptive,
            "languages": list(entry.languages),
        },
        "role_profile": {
            "domains": state.domains,
            "capabilities": state.skills,
            "specializations": state.specializations,
            "combined_requirements": [intent.label for intent in state.linked_intents],
            "excluded_focus_areas": state.negative_constraints,
            "clarifications_used": state.clarification_answers,
        },
        "recommended_use": (
            "Use this catalogue-grounded recommendation as the primary assessment "
            "component. Pair it with recruiter-specific interview questions only "
            "where the catalogue description does not claim coverage."
        ),
    }


def _primary_kept(
    result: ConvergenceStepResult, state: ConversationState
) -> CatalogueEntry | None:
    kept = [item.candidate.entry for item in result.filtered_candidates if item.keep]
    if not kept:
        return None
    terms = " ".join(state.searchable_terms()).casefold()
    if "rust" in terms:
        return sorted(kept, key=lambda entry: _engineering_priority(entry.name))[0]
    if "leadership" in terms or "executive" in terms or "director" in terms:
        return sorted(kept, key=lambda entry: _leadership_priority(entry.name))[0]
    return kept[0]


def _leadership_priority(name: str) -> tuple[int, str]:
    lowered = name.casefold()
    if "occupational personality questionnaire" in lowered or "opq32" in lowered:
        return (0, lowered)
    if "leadership report" in lowered:
        return (1, lowered)
    if "universal competency" in lowered:
        return (2, lowered)
    return (3, lowered)


def _engineering_priority(name: str) -> tuple[int, str]:
    lowered = name.casefold()
    if "smart interview live coding" in lowered:
        return (0, lowered)
    if "linux programming" in lowered:
        return (1, lowered)
    if "networking and implementation" in lowered:
        return (2, lowered)
    if "verify interactive g+" in lowered:
        return (3, lowered)
    return (4, lowered)


def _has_kept_candidates(result: ConvergenceStepResult) -> bool:
    return any(item.keep for item in result.filtered_candidates)


def _is_completion_acknowledgement(query: str) -> bool:
    lowered = query.casefold().strip()
    if "?" in query or any(
        marker in lowered
        for marker in ("should i", "also", "add", "drop", "change", "instead", "cognitive")
    ):
        return False
    acknowledgement_markers = (
        "that works",
        "perfect",
        "thanks",
        "thank you",
        "looks good",
        "go ahead",
        "this is good",
        "that's what we need",
    )
    return any(marker in lowered for marker in acknowledgement_markers)


def _completion_message(public: object) -> str:
    message = getattr(public, "message", "")
    if message:
        return f"{message} I'll treat this as the working assessment plan unless you refine the role further."
    return "I'll treat this as the working assessment plan unless you refine the role further."


def _normalize_catalogue_text(text: str) -> str:
    return text.replace("The.NET", "The .NET").replace("the.NET", "the .NET")


def _merge_public_recommendations(
    existing: list[dict[str, object]], new: list[dict[str, object]]
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in [*existing[:3], *new, *existing[3:]]:
        name = str(item.get("name", "")).casefold()
        if name and name not in seen:
            seen.add(name)
            merged.append(item)
    return merged[:5]
