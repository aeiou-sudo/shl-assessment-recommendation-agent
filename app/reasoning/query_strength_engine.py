"""Analyze whether the positive query state is strong enough to converge."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.core.config import settings
from app.models.catalog import FilteredCandidate
from app.reasoning.confidence_engine import ConfidenceReport


@dataclass(frozen=True)
class QueryStrengthReport:
    strong_enough: bool
    shared_terms: list[str]
    reason: str


class QueryStrengthEngine:
    def analyze(
        self,
        filtered: list[FilteredCandidate],
        confidence: ConfidenceReport,
    ) -> QueryStrengthReport:
        kept = [item.candidate for item in filtered if item.keep]
        if not kept:
            return QueryStrengthReport(False, [], "No candidates survived filtering.")
        shared_terms = _shared_terms(kept[:5])
        strong = (
            confidence.top_score >= settings.weak_query_threshold
            and confidence.entropy < 0.86
        )
        reason = (
            "Query has enough signal for ambiguity analysis."
            if strong
            else "Query is weak; strengthen around recurring retrieved concepts first."
        )
        return QueryStrengthReport(strong, shared_terms, reason)


def _shared_terms(candidates: list[object]) -> list[str]:
    counter: Counter[str] = Counter()
    for candidate in candidates:
        text = f"{candidate.entry.name} {candidate.entry.description} {' '.join(candidate.entry.keys)}"
        seen = {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{2,}", text.casefold())
            if token not in STOP_WORDS
        }
        counter.update(seen)
    return [term for term, count in counter.most_common(8) if count >= 2]


STOP_WORDS = {
    "the",
    "and",
    "for",
    "this",
    "that",
    "with",
    "test",
    "measures",
    "knowledge",
    "assessment",
    "skills",
    "new",
    "multiple",
    "choice",
    "multi",
    "designed",
}
