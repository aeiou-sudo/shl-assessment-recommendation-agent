"""Build the positive semantic search query from state."""

from __future__ import annotations

from app.state.conversation_state import ConversationState


class QuerySynthesizer:
    """Negative constraints are deliberately excluded from this query."""

    def synthesize(self, state: ConversationState) -> str:
        terms = state.searchable_terms()
        anchors = _anchor_terms(terms)
        return "\n".join(_dedupe([*anchors, *terms]))


def _anchor_terms(terms: list[str]) -> list[str]:
    joined = " ".join(terms).casefold()
    anchors: list[str] = []
    if any(term in joined for term in ("leadership", "executive", "director", "cxo")):
        anchors.extend(
            [
                "Occupational Personality Questionnaire OPQ32r",
                "OPQ Leadership Report",
                "OPQ Universal Competency Report",
                "personality behavior leadership selection benchmark",
            ]
        )
    if any(term in joined for term in ("cognitive", "reasoning", "aptitude")):
        anchors.append("SHL Verify Interactive G+ ability aptitude reasoning")
    if any(term in joined for term in ("live coding", "coding interview")):
        anchors.append("Smart Interview Live Coding")
    if "rust" in joined and any(
        term in joined for term in ("networking", "infrastructure", "systems", "performance")
    ):
        anchors.extend(
            [
                "Smart Interview Live Coding",
                "Linux Programming General",
                "Networking and Implementation",
                "systems programming infrastructure implementation",
            ]
        )
    return anchors


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            output.append(value)
    return output
