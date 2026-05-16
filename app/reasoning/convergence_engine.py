"""Orchestrate one staged convergence step."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.models.catalog import FilteredCandidate
from app.reasoning.analyze_candidates_enriched import (
    AmbiguityReport,
    CandidateAmbiguityAnalyzer,
)
from app.reasoning.confidence_engine import ConfidenceEngine, ConfidenceReport
from app.reasoning.intent_filters import IntentFilter
from app.reasoning.llm_clarification_engine import ClarificationEngine
from app.reasoning.query_gap_analyzer import QueryGapAnalyzer
from app.reasoning.query_strength_engine import QueryStrengthEngine, QueryStrengthReport
from app.reasoning.trajectory_engine import TrajectoryEngine, TrajectoryReport
from app.retrieval.semantic_search import SemanticSearch
from app.state.conversation_state import ConfidenceSnapshot, ConversationState, QueryType
from app.state.state_manager import StateManager


class StepStatus(StrEnum):
    REJECTED = "REJECTED"
    NEEDS_QUERY_STRENGTHENING = "NEEDS_QUERY_STRENGTHENING"
    NEEDS_AMBIGUITY_CLARIFICATION = "NEEDS_AMBIGUITY_CLARIFICATION"
    CONVERGED = "CONVERGED"
    NO_EXACT_MATCH = "NO_EXACT_MATCH"


@dataclass(frozen=True)
class ConvergenceStepResult:
    status: StepStatus
    message: str
    filtered_candidates: list[FilteredCandidate] = field(default_factory=list)
    confidence: ConfidenceReport | None = None
    strength: QueryStrengthReport | None = None
    ambiguity: AmbiguityReport | None = None
    trajectory: TrajectoryReport | None = None


class ConvergenceEngine:
    def __init__(
        self,
        search: SemanticSearch,
        state_manager: StateManager | None = None,
        intent_filter: IntentFilter | None = None,
        confidence_engine: ConfidenceEngine | None = None,
        strength_engine: QueryStrengthEngine | None = None,
        gap_analyzer: QueryGapAnalyzer | None = None,
        ambiguity_analyzer: CandidateAmbiguityAnalyzer | None = None,
        clarification_engine: ClarificationEngine | None = None,
        trajectory_engine: TrajectoryEngine | None = None,
    ) -> None:
        self.search = search
        self.state_manager = state_manager or StateManager()
        self.intent_filter = intent_filter or IntentFilter()
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.strength_engine = strength_engine or QueryStrengthEngine()
        self.gap_analyzer = gap_analyzer or QueryGapAnalyzer()
        self.ambiguity_analyzer = ambiguity_analyzer or CandidateAmbiguityAnalyzer()
        self.clarification_engine = clarification_engine or ClarificationEngine()
        self.trajectory_engine = trajectory_engine or TrajectoryEngine()

    def run_turn(self, query: str, state: ConversationState) -> ConvergenceStepResult:
        interpreted = self.state_manager.apply_user_turn(query, state)
        if interpreted.query_type == QueryType.OUT_OF_CONTEXT:
            return ConvergenceStepResult(
                status=StepStatus.REJECTED,
                message=(
                    "I can help with hiring assessment preparation and catalogue-based "
                    "assessment planning. Please ask about a role, skill, domain, or "
                    "candidate assessment need."
                ),
            )

        retrieved = self.search.retrieve(state)
        filtered = self.intent_filter.filter(retrieved, state)
        confidence = self.confidence_engine.evaluate(filtered)
        state.confidence_history.append(
            ConfidenceSnapshot(
                top_score=confidence.top_score,
                spread=confidence.spread,
                entropy=confidence.entropy,
                reason=confidence.reason,
            )
        )
        strength = self.strength_engine.analyze(filtered, confidence)
        trajectory = self.trajectory_engine.evaluate(state)

        kept = [item for item in filtered if item.keep]
        if not kept:
            return ConvergenceStepResult(
                status=StepStatus.NO_EXACT_MATCH,
                message="I could not find a viable catalogue match yet. Can you name the core role or skill?",
                filtered_candidates=filtered,
                confidence=confidence,
                strength=strength,
                trajectory=trajectory,
            )

        if confidence.converged:
            return ConvergenceStepResult(
                status=StepStatus.CONVERGED,
                message="Converged on a catalogue entry.",
                filtered_candidates=filtered,
                confidence=confidence,
                strength=strength,
                trajectory=trajectory,
            )

        if _has_stable_catalogue_family(state, filtered, confidence):
            return ConvergenceStepResult(
                status=StepStatus.CONVERGED,
                message="Converged on a stable catalogue family.",
                filtered_candidates=filtered,
                confidence=confidence,
                strength=strength,
                trajectory=trajectory,
            )

        if not strength.strong_enough:
            question = self.gap_analyzer.question(state, filtered, strength)
            state.ambiguity_history.append(question)
            return ConvergenceStepResult(
                status=StepStatus.NEEDS_QUERY_STRENGTHENING,
                message=self.clarification_engine.from_strength_gap(question, strength),
                filtered_candidates=filtered,
                confidence=confidence,
                strength=strength,
                trajectory=trajectory,
            )

        ambiguity = self.ambiguity_analyzer.analyze(state, filtered)
        if ambiguity.ambiguous and ambiguity.question:
            state.ambiguity_history.append(ambiguity.question)
            return ConvergenceStepResult(
                status=StepStatus.NEEDS_AMBIGUITY_CLARIFICATION,
                message=self.clarification_engine.from_ambiguity(ambiguity),
                filtered_candidates=filtered,
                confidence=confidence,
                strength=strength,
                ambiguity=ambiguity,
                trajectory=trajectory,
            )

        return ConvergenceStepResult(
            status=StepStatus.NO_EXACT_MATCH,
            message=(
                "I found related catalogue entries, but no exact match is separated yet. "
                f"The closest available option is {kept[0].candidate.entry.name}."
            ),
            filtered_candidates=filtered,
            confidence=confidence,
            strength=strength,
            ambiguity=ambiguity,
            trajectory=trajectory,
        )


def _has_stable_catalogue_family(
    state: ConversationState,
    filtered: list[FilteredCandidate],
    confidence: ConfidenceReport,
) -> bool:
    kept = [item.candidate for item in filtered if item.keep][:8]
    if confidence.top_score < 0.70 or len(kept) < 2:
        return False

    terms = " ".join(state.searchable_terms()).casefold()
    engineering_stack = (
        state.turns >= 2
        and "rust" in terms
        and any(term in terms for term in ("networking", "infrastructure", "systems", "implementation"))
        and _has_engineering_stack_matches(kept)
    )
    if engineering_stack:
        return True

    if state.turns < 3 or confidence.top_score < 0.78:
        return False

    has_clear_purpose = any(
        term in terms
        for term in (
            "selection",
            "benchmark",
            "development",
            "promotion",
            "hiring",
            "senior",
            "executive",
            "director",
            "leadership",
        )
    )
    if not has_clear_purpose:
        return False

    names = [item.entry.name.casefold() for item in kept]
    categories = [" ".join(item.entry.keys).casefold() for item in kept]
    opq_family = sum("opq" in name or "occupational personality" in name for name in names)
    leadership_family = sum("leadership" in name for name in names)
    category_counts = [categories.count(category) for category in set(categories) if category]
    shared_category = max(category_counts) if category_counts else 0
    return opq_family >= 2 or leadership_family >= 2 or shared_category >= 4


def _has_engineering_stack_matches(kept: list[object]) -> bool:
    names = " ".join(item.entry.name.casefold() for item in kept[:6])
    signals = sum(
        signal in names
        for signal in ("smart interview", "linux programming", "networking and implementation")
    )
    return signals >= 2
