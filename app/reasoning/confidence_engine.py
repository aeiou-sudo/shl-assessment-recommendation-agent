"""Confidence and entropy calculations over filtered retrievals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.config import settings
from app.models.catalog import FilteredCandidate


@dataclass(frozen=True)
class ConfidenceReport:
    top_score: float
    second_score: float
    spread: float
    entropy: float
    confidence: float
    converged: bool
    reason: str


class ConfidenceEngine:
    def evaluate(self, filtered: list[FilteredCandidate]) -> ConfidenceReport:
        kept = [item.candidate for item in filtered if item.keep]
        if not kept:
            return ConfidenceReport(0.0, 0.0, 0.0, 1.0, 0.0, False, "No kept candidates.")

        scores = [max(0.0, item.score) for item in kept]
        top = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        spread = top - second
        entropy = _entropy(scores)
        confidence = _clamp((top * 0.62) + (spread * 0.32) + ((1.0 - entropy) * 0.22))
        converged = (
            confidence >= settings.convergence_threshold
            and spread >= settings.ambiguity_margin
        )
        reason = (
            "Top candidate is sufficiently separated."
            if converged
            else "Retrieval distribution remains weak or ambiguous."
        )
        return ConfidenceReport(top, second, spread, entropy, confidence, converged, reason)


def _entropy(scores: list[float]) -> float:
    if not scores:
        return 1.0
    total = sum(scores)
    if total <= 0:
        return 1.0
    probs = [score / total for score in scores if score > 0]
    value = -sum(prob * math.log(prob) for prob in probs)
    max_entropy = math.log(len(scores)) if len(scores) > 1 else 1.0
    return _clamp(value / max_entropy)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
