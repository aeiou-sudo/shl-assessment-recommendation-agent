"""Classify and extract state updates from a user turn."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.llm.groq_client import GroqReasoningClient
from app.state.conversation_state import ConversationState, QueryType


HIRING_TERMS = {
    "hire",
    "hiring",
    "candidate",
    "assessment",
    "test",
    "role",
    "job",
    "skill",
    "skills",
    "developer",
    "engineer",
    "manager",
    "analyst",
    "sales",
    "support",
    "interview",
    "recruiter",
    "aptitude",
    "personality",
    "cognitive",
    "language",
    "leadership",
    "leader",
    "senior",
    "executive",
    "director",
    "selection",
    "promotion",
    "development",
    "position",
    "positions",
}

NEGATIVE_PATTERNS = [
    r"\bnot\s+([a-zA-Z0-9+#.\- ]{2,40})",
    r"\bwithout\s+([a-zA-Z0-9+#.\- ]{2,40})",
    r"\bexclude\s+([a-zA-Z0-9+#.\- ]{2,40})",
    r"\bavoid\s+([a-zA-Z0-9+#.\- ]{2,40})",
    r"\bno\s+([a-zA-Z0-9+#.\- ]{2,40})",
]


@dataclass
class InterpretedQuery:
    query_type: QueryType
    domains: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    specializations: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    raw_phrases: list[str] = field(default_factory=list)
    rationale: str = ""


class LLMStateInterpreter:
    """Constrained reasoning stage for query extraction and classification."""

    def __init__(self, llm: GroqReasoningClient | None = None) -> None:
        self.llm = llm or GroqReasoningClient()

    def interpret(self, query: str, state: ConversationState) -> InterpretedQuery:
        fallback = self._fallback_interpret(query, state)
        payload = {
            "query": query,
            "state": {
                "primary_intents": [intent.label for intent in state.primary_intents],
                "linked_intents": [intent.label for intent in state.linked_intents],
                "domains": state.domains,
                "skills": state.skills,
                "specializations": state.specializations,
                "rejected_intents": state.rejected_intents,
                "ambiguity_history": state.ambiguity_history[-5:],
                "clarification_answers": state.clarification_answers[-5:],
                "negative_constraints": state.negative_constraints,
            },
            "allowed_query_types": [item.value for item in QueryType],
        }
        system = (
            "You classify HR assessment-preparation queries only. Return JSON with "
            "query_type, domains, skills, specializations, negative_constraints, "
            "raw_phrases, rationale. Reject non-hiring or non-assessment requests as "
            "OUT_OF_CONTEXT. Extract only explicit or strongly implied user intent. "
            "Do not invent catalogue entries."
        )
        parsed = self.llm.json_call(system, payload, fallback.__dict__)
        try:
            query_type = QueryType(parsed.get("query_type", fallback.query_type.value))
        except ValueError:
            query_type = fallback.query_type
        return InterpretedQuery(
            query_type=query_type,
            domains=_clean_list(parsed.get("domains", fallback.domains)),
            skills=_clean_list(parsed.get("skills", fallback.skills)),
            specializations=_clean_list(
                parsed.get("specializations", fallback.specializations)
            ),
            negative_constraints=_clean_list(
                parsed.get("negative_constraints", fallback.negative_constraints)
            ),
            raw_phrases=_clean_list(parsed.get("raw_phrases", fallback.raw_phrases)),
            rationale=str(parsed.get("rationale", fallback.rationale)),
        )

    def _fallback_interpret(self, query: str, state: ConversationState) -> InterpretedQuery:
        lowered = query.casefold()
        terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}", lowered))
        has_hiring_signal = bool(terms & HIRING_TERMS)
        is_short_clarification = state.ambiguity_history and len(terms) <= 8
        if not has_hiring_signal and not is_short_clarification:
            return InterpretedQuery(
                query_type=QueryType.OUT_OF_CONTEXT,
                rationale="No hiring or assessment-preparation signal was detected.",
            )

        negatives: list[str] = []
        for pattern in NEGATIVE_PATTERNS:
            negatives.extend(match.strip(" .") for match in re.findall(pattern, query, re.I))

        query_type = QueryType.NEW_QUERY
        if is_short_clarification:
            query_type = QueryType.CLARIFICATION_RESPONSE
        elif state.primary_intents and any(
            word in lowered for word in ("instead", "change", "switch", "actually")
        ):
            query_type = QueryType.INTENT_SHIFT
        elif state.primary_intents and any(
            word in lowered for word in ("also", "with", "but", "prefer", "need", "exclude")
        ):
            query_type = QueryType.REFINEMENT_QUERY

        extracted = _extract_positive_phrases(query, negatives)
        return InterpretedQuery(
            query_type=query_type,
            skills=extracted,
            negative_constraints=negatives,
            raw_phrases=[query],
            rationale="Deterministic fallback extraction.",
        )


def _extract_positive_phrases(query: str, negatives: list[str]) -> list[str]:
    text = query
    for negative in negatives:
        text = re.sub(re.escape(negative), " ", text, flags=re.I)
    stop = {
        "hire",
        "hiring",
        "candidate",
        "candidates",
        "assessment",
        "assessments",
        "test",
        "tests",
        "role",
        "roles",
        "job",
        "jobs",
        "recruiter",
        "interview",
        "solution",
        "solutions",
        "pool",
        "consists",
        "people",
        "years",
        "experience",
        "position",
        "positions",
        "of",
        "than",
        "more",
        "less",
        "against",
        "comparing",
        "compare",
        "meant",
        "intended",
        "use",
        "for",
        "and",
        "with",
        "who",
        "what",
        "can",
        "should",
        "need",
        "needs",
        "needed",
        "we",
        "our",
        "their",
        "looking",
        "prepare",
        "plan",
        "an",
        "a",
        "the",
        "to",
        "not",
        "no",
        "without",
        "exclude",
        "avoid",
        "yes",
        "go",
        "ahead",
        "also",
        "add",
        "this",
        "level",
    }
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}", text):
        cleaned = token.strip(" .;,")
        if cleaned and cleaned.casefold() not in stop:
            tokens.append(cleaned)
    phrases = [" ".join(tokens)] if tokens else []
    phrases.extend(tokens[:8])
    return _clean_list(phrases)


def _clean_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip(" .;,")
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result
