"""Conservative LLM filtering over deterministic retrieval results."""

from __future__ import annotations

import re

from app.llm.groq_client import GroqReasoningClient
from app.models.catalog import FilteredCandidate, RetrievalCandidate
from app.state.conversation_state import ConversationState


class IntentFilter:
    def __init__(self, llm: GroqReasoningClient | None = None) -> None:
        self.llm = llm or GroqReasoningClient()

    def filter(
        self,
        candidates: list[RetrievalCandidate],
        state: ConversationState,
    ) -> list[FilteredCandidate]:
        if not candidates:
            return []
        fallback = self._heuristic_filter(candidates, state)
        payload = {
            "positive_intent_terms": state.searchable_terms(),
            "negative_constraints": state.negative_constraints,
            "candidates": [
                {
                    "entity_id": item.entry.entity_id,
                    "name": item.entry.name,
                    "description": item.entry.description,
                    "keys": list(item.entry.keys),
                    "score": item.score,
                }
                for item in candidates
            ],
        }
        system = (
            "You conservatively filter SHL catalogue retrieval candidates. Return JSON "
            "with items: [{entity_id, keep, reason, violates_negative_constraints}]. "
            "Only remove clear noise, domain drift, or entries that violate explicit "
            "negative constraints. Preserve uncertain but plausible matches. Do not "
            "invent or rewrite catalogue content."
        )
        parsed = self.llm.json_call(
            system,
            payload,
            {
                "items": [
                    {
                        "entity_id": item.candidate.entry.entity_id,
                        "keep": item.keep,
                        "reason": item.reason,
                        "violates_negative_constraints": item.violates_negative_constraints,
                    }
                    for item in fallback
                ]
            },
        )
        by_id = {
            str(item.get("entity_id")): item
            for item in parsed.get("items", [])
            if isinstance(item, dict)
        }
        filtered: list[FilteredCandidate] = []
        for candidate in candidates:
            item = by_id.get(candidate.entry.entity_id)
            if item is None:
                filtered.append(
                    FilteredCandidate(candidate, keep=True, reason="No LLM decision; preserved.")
                )
                continue
            filtered.append(
                FilteredCandidate(
                    candidate=candidate,
                    keep=bool(item.get("keep", True)),
                    reason=str(item.get("reason", "Conservative filtering decision.")),
                    violates_negative_constraints=bool(
                        item.get("violates_negative_constraints", False)
                    ),
                )
            )
        return filtered

    def _heuristic_filter(
        self, candidates: list[RetrievalCandidate], state: ConversationState
    ) -> list[FilteredCandidate]:
        constraints = [item.casefold() for item in state.negative_constraints]
        output: list[FilteredCandidate] = []
        for candidate in candidates:
            haystack = " ".join(
                [
                    candidate.entry.name,
                    candidate.entry.description,
                    " ".join(candidate.entry.keys),
                ]
            ).casefold()
            violation = any(_contains_phrase(haystack, constraint) for constraint in constraints)
            keep = not violation and candidate.score > 0.05
            reason = (
                "Violates an explicit negative constraint."
                if violation
                else "Kept as a plausible deterministic retrieval."
            )
            output.append(
                FilteredCandidate(
                    candidate=candidate,
                    keep=keep,
                    reason=reason,
                    violates_negative_constraints=violation,
                )
            )
        return output


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.strip()
    if not phrase:
        return False
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))
