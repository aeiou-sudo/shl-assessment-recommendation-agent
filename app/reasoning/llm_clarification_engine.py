"""Clarification response helpers."""

from __future__ import annotations

from app.reasoning.analyze_candidates_enriched import AmbiguityReport
from app.reasoning.query_strength_engine import QueryStrengthReport


class ClarificationEngine:
    def from_strength_gap(self, question: str, strength: QueryStrengthReport) -> str:
        return question

    def from_ambiguity(self, report: AmbiguityReport) -> str:
        return report.question
