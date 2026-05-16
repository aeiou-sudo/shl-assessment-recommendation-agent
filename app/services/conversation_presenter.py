"""Recruiter-facing conversation rendering.

This module is deliberately outside retrieval and reasoning. The convergence
engine can keep backend statuses, scores, entropy, and candidate filtering
details; this layer turns the result into consultative recruiter guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.catalog import CatalogueEntry
from app.reasoning.convergence_engine import ConvergenceStepResult, StepStatus
from app.state.conversation_state import ConversationState


@dataclass(frozen=True)
class PublicRecommendation:
    name: str
    category: str
    relevance: str
    duration: str
    url: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "relevance": self.relevance,
            "duration": self.duration or "Not specified",
            "url": self.url,
        }


@dataclass(frozen=True)
class PublicConversation:
    message: str
    recommendations: list[dict[str, object]] = field(default_factory=list)
    assessment_plan: dict[str, object] | None = None
    end_of_conversation: bool = False


class ConversationPresenter:
    """Hide orchestration mechanics and speak as an assessment consultant."""

    def render(
        self,
        result: ConvergenceStepResult,
        state: ConversationState,
        assessment_plan: dict[str, object] | None,
    ) -> PublicConversation:
        if result.status == StepStatus.REJECTED:
            return PublicConversation(
                message=(
                    "I can help with assessment planning for hiring, selection, "
                    "promotion, or development decisions. Tell me the role, level, "
                    "and the capabilities you want to evaluate."
                )
            )

        if result.status in {
            StepStatus.NEEDS_QUERY_STRENGTHENING,
            StepStatus.NEEDS_AMBIGUITY_CLARIFICATION,
        }:
            return PublicConversation(
                message=self._clarification_message(result.message, state)
            )

        if result.status == StepStatus.CONVERGED and assessment_plan:
            recommendations = [
                self._recommendation(item.candidate.entry, state).as_dict()
                for item in _ordered_kept(result, state)
            ][:_recommendation_limit(state)]
            message = self._recommendation_message(recommendations, state)
            return PublicConversation(
                message=message,
                recommendations=recommendations,
                assessment_plan=assessment_plan,
                end_of_conversation=False,
            )

        if result.status == StepStatus.NO_EXACT_MATCH:
            alternatives = [
                self._recommendation(item.candidate.entry, state).as_dict()
                for item in _ordered_kept(result, state)
            ][:_recommendation_limit(state)]
            if alternatives:
                return PublicConversation(
                    message=(
                        "I do not see a dedicated catalogue match for that exact "
                        "combination. The closest workable options are below; together "
                        "they cover the strongest parts of the role profile you described."
                    ),
                    recommendations=alternatives,
                    assessment_plan=assessment_plan,
                    end_of_conversation=False,
                )
            return PublicConversation(
                message=(
                    "I do not have enough role signal yet to make a reliable assessment "
                    "recommendation. What is the core role or capability you need to evaluate?"
                )
            )

        return PublicConversation(
            message="Could you share a little more about the role and assessment purpose?"
        )

    def render_completion(self, state: ConversationState) -> PublicConversation:
        return PublicConversation(
            message=state.final_message
            or "Great, I will keep this assessment plan as the final recommendation.",
            recommendations=state.final_recommendations,
            assessment_plan=state.final_assessment_plan,
            end_of_conversation=True,
        )

    def _clarification_message(self, question: str, state: ConversationState) -> str:
        clean_question = _consultative_question(question, state)
        memory = _memory_clause(state)
        if memory:
            return f"{memory} {clean_question}"
        return clean_question

    def _recommendation_message(
        self, recommendations: list[dict[str, object]], state: ConversationState
    ) -> str:
        if not recommendations:
            return "I have enough to recommend an assessment plan."
        first = recommendations[0]["name"]
        memory = _memory_clause(state)
        if memory:
            if _has_unavailable_specific_stack(state, recommendations):
                return (
                    f"{memory} I do not see a dedicated catalogue assessment for that "
                    f"specific technology stack, so I would use {first} as the primary "
                    "fit and combine it with the closest supporting assessments below."
                )
            return (
                f"{memory} Based on that, I would use {first} as the primary fit. "
                "The shortlist below keeps the plan grounded in the catalogue while "
                "covering the role requirements you described."
            )
        if _has_unavailable_specific_stack(state, recommendations):
            return (
                "I do not see a dedicated catalogue assessment for that exact technology "
                f"stack, so I would use {first} as the primary fit and combine it with "
                "the closest supporting assessments below."
            )
        return (
            f"I would use {first} as the primary fit. The shortlist below is grounded "
            "in the catalogue and aligned to the role requirements you described."
        )

    def _recommendation(
        self, entry: CatalogueEntry, state: ConversationState
    ) -> PublicRecommendation:
        category = ", ".join(entry.keys) if entry.keys else "Catalogue assessment"
        focus = _focus_phrase(state)
        description = _normalize_catalogue_text(entry.description)
        relevance = (
            f"Fits the {focus} profile and evaluates: {description}"
            if focus
            else f"Matches the assessment need described and evaluates: {description}"
        )
        return PublicRecommendation(
            name=entry.name,
            category=category,
            relevance=_truncate(relevance, 260),
            duration=entry.duration,
            url=entry.link,
        )


def _consultative_question(question: str, state: ConversationState) -> str:
    lowered = question.casefold()
    banned = (
        "semantic",
        "retrieval",
        "query",
        "candidate vectors",
        "confidence",
        "clusters",
        "keywords",
    )
    if any(term in lowered for term in banned):
        return _fallback_question(state)
    if question.strip().endswith("?"):
        return question.strip()
    return _fallback_question(state)


def _ordered_kept(
    result: ConvergenceStepResult, state: ConversationState
) -> list[object]:
    kept = [item for item in result.filtered_candidates if item.keep]
    terms = " ".join(state.searchable_terms()).casefold()
    if "rust" in terms:
        return sorted(kept, key=lambda item: _engineering_priority(item.candidate.entry.name))
    if "leadership" in terms or "executive" in terms or "director" in terms:
        return sorted(kept, key=lambda item: _leadership_priority(item.candidate.entry.name))
    return kept


def _recommendation_limit(state: ConversationState) -> int:
    terms = " ".join(state.searchable_terms()).casefold()
    if "cognitive" in terms or "reasoning" in terms:
        return 5
    return 3


def _has_unavailable_specific_stack(
    state: ConversationState, recommendations: list[dict[str, object]]
) -> bool:
    terms = " ".join(state.searchable_terms()).casefold()
    if "rust" not in terms:
        return False
    names = " ".join(str(item.get("name", "")) for item in recommendations).casefold()
    return "rust" not in names


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


def _fallback_question(state: ConversationState) -> str:
    terms = [item.casefold() for item in state.searchable_terms()]
    joined = " ".join(terms)
    if any(term in joined for term in ("leadership", "director", "executive", "manager")):
        return (
            "Is this assessment mainly for selection against a leadership benchmark, "
            "or for development feedback for someone already in role?"
        )
    if any(term in joined for term in ("data", "analytics", "machine", "ml", "model")):
        return (
            "Should the role focus more on data analysis, predictive modeling, "
            "or production ML deployment?"
        )
    if any(term in joined for term in ("security", "secure", "cyber")):
        return (
            "Is the focus secure infrastructure engineering, application security, "
            "or security analysis within a broader technical role?"
        )
    if any(term in joined for term in ("engineer", "developer", "software", "backend", "net")):
        return (
            "Would this role require hands-on application development, infrastructure "
            "ownership, or broader technical architecture?"
        )
    return (
        "What would success in this role depend on most: technical specialization, "
        "problem solving, people leadership, or communication with stakeholders?"
    )


def _memory_clause(state: ConversationState) -> str:
    pieces: list[str] = []
    if state.specializations:
        pieces.append(f"the {_join_naturally(_compact_terms(state.specializations))} specialization")
    elif state.skills:
        pieces.append(f"the {_join_naturally(_compact_terms(state.skills))} focus")
    if state.domains:
        pieces.append(f"the {_join_naturally(_compact_terms(state.domains))} domain")
    if state.negative_constraints:
        pieces.append(f"excluding {_join_naturally(_compact_terms(state.negative_constraints))}")
    if not pieces:
        return ""
    return f"I'm keeping {_join_naturally(pieces)} in view."


def _focus_phrase(state: ConversationState) -> str:
    for values in (state.specializations, state.skills, state.domains):
        if values:
            return ", ".join(_compact_terms(values)[:3])
    if state.primary_intents:
        return state.primary_intents[0].label
    return ""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    sentence_end = text.rfind(".", 0, limit)
    if sentence_end > int(limit * 0.55):
        return text[: sentence_end + 1]
    truncated = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    while truncated.rsplit(" ", 1)[-1].casefold() in {
        "the",
        "and",
        "or",
        "with",
        "of",
        "in",
        "includes",
        "include",
    }:
        truncated = truncated.rsplit(" ", 1)[0].rstrip(" ,;:")
    return truncated + "."


def _join_naturally(values: list[str]) -> str:
    if len(values) <= 1:
        return values[0] if values else ""
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _compact_terms(values: list[str]) -> list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _pretty_term(value)
        key = text.casefold()
        if not key or key in seen:
            continue
        if any(_is_subterm(key, existing.casefold()) for existing in compacted):
            continue
        compacted.append(text)
        seen.add(key)
        if len(compacted) == 2:
            break
    return compacted


def _is_subterm(term: str, existing: str) -> bool:
    return term != existing and term in existing


def _pretty_term(value: str) -> str:
    words = value.strip().split()
    pretty = [".NET" if word.upper() == "NET" else word for word in words]
    return " ".join(pretty)


def _normalize_catalogue_text(text: str) -> str:
    return text.replace("The.NET", "The .NET").replace("the.NET", "the .NET")
