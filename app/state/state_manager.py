"""Conversation state transition logic."""

from __future__ import annotations

from app.state.conversation_state import ConversationState, IntentState, QueryType
from app.state.llm_state_interpreter import InterpretedQuery, LLMStateInterpreter


class StateManager:
    def __init__(self, interpreter: LLMStateInterpreter | None = None) -> None:
        self.interpreter = interpreter or LLMStateInterpreter()

    def apply_user_turn(self, query: str, state: ConversationState) -> InterpretedQuery:
        interpreted = self.interpreter.interpret(query, state)
        state.turns += 1
        state.last_query_type = interpreted.query_type
        state.last_user_query = query

        if interpreted.query_type == QueryType.OUT_OF_CONTEXT:
            state.rejected_intents.append(query)
            return interpreted

        if interpreted.query_type == QueryType.INTENT_SHIFT:
            for intent in state.primary_intents:
                intent.active = False
            state.primary_intents = []
            state.domains = []
            state.skills = []
            state.specializations = []

        target = state.primary_intents
        if interpreted.query_type == QueryType.REFINEMENT_QUERY and state.primary_intents:
            target = state.linked_intents if _looks_linked(interpreted) else state.primary_intents

        label = _best_label(query, interpreted)
        if not state.primary_intents:
            state.primary_intents.append(
                IntentState(
                    label=label,
                    domains=interpreted.domains,
                    skills=interpreted.skills,
                    specializations=interpreted.specializations,
                    raw_phrases=interpreted.raw_phrases or [query],
                )
            )
        elif target is state.linked_intents:
            state.linked_intents.append(
                IntentState(
                    label=label,
                    domains=interpreted.domains,
                    skills=interpreted.skills,
                    specializations=interpreted.specializations,
                    raw_phrases=interpreted.raw_phrases or [query],
                )
            )
        else:
            intent = state.primary_intents[0]
            intent.domains = _merge(intent.domains, interpreted.domains)
            intent.skills = _merge(intent.skills, interpreted.skills)
            intent.specializations = _merge(
                intent.specializations, interpreted.specializations
            )
            intent.raw_phrases = _merge(intent.raw_phrases, interpreted.raw_phrases or [query])

        state.domains = _merge(state.domains, interpreted.domains)
        state.skills = _merge(state.skills, interpreted.skills)
        state.specializations = _merge(state.specializations, interpreted.specializations)
        state.negative_constraints = _merge(
            state.negative_constraints, interpreted.negative_constraints
        )
        if interpreted.query_type == QueryType.CLARIFICATION_RESPONSE:
            state.clarification_answers.append(query)
        return interpreted


def _looks_linked(interpreted: InterpretedQuery) -> bool:
    phrases = " ".join(interpreted.raw_phrases).casefold()
    return any(word in phrases for word in ("also", "plus", "and", "combined", "linked"))


def _best_label(query: str, interpreted: InterpretedQuery) -> str:
    for collection in (
        interpreted.specializations,
        interpreted.skills,
        interpreted.domains,
        interpreted.raw_phrases,
    ):
        if collection:
            return collection[0]
    return query.strip()


def _merge(left: list[str], right: list[str]) -> list[str]:
    seen = {item.casefold(): item for item in left if item.strip()}
    for item in right:
        key = item.casefold().strip()
        if key and key not in seen:
            seen[key] = item.strip()
    return list(seen.values())
