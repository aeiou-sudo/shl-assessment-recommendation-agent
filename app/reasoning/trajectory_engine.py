"""Track convergence trajectory across turns."""

from __future__ import annotations

from dataclasses import dataclass

from app.state.conversation_state import ConversationState


@dataclass(frozen=True)
class TrajectoryReport:
    improving: bool
    stalled: bool
    reason: str


class TrajectoryEngine:
    def evaluate(self, state: ConversationState) -> TrajectoryReport:
        history = state.confidence_history[-3:]
        if len(history) < 2:
            return TrajectoryReport(True, False, "Insufficient history; continue.")
        improving = history[-1].top_score >= history[0].top_score
        stalled = len(history) >= 3 and max(item.spread for item in history) < 0.04
        reason = (
            "Confidence is improving."
            if improving and not stalled
            else "Confidence is not separating candidates."
        )
        return TrajectoryReport(improving, stalled, reason)
