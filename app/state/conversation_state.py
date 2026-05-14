from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConversationState:

    # -----------------------------------------
    # Role / specialization
    # -----------------------------------------

    primary_intent: str = None

    intent_history: List[str] = field(
        default_factory=list
    )

    role_focus: Optional[str] = None

    domains: List[str] = field(
        default_factory=list
    )

    excluded_roles: List[str] = field(
        default_factory=list
    )

    excluded_domains: List[str] = field(
        default_factory=list
    )

    # -----------------------------------------
    # Technologies
    # -----------------------------------------

    technology_stack: List[str] = field(
        default_factory=list
    )

    excluded_technologies: List[str] = field(
        default_factory=list
    )

    # -----------------------------------------
    # Competencies
    # -----------------------------------------

    competencies: List[str] = field(
        default_factory=list
    )

    # -----------------------------------------
    # Seniority
    # -----------------------------------------

    seniority: Optional[str] = None

    # -----------------------------------------
    # Role modifiers
    # -----------------------------------------

    include_devops: Optional[bool] = None

    include_frontend: Optional[bool] = None

    # -----------------------------------------
    # Conversation metadata
    # -----------------------------------------

    clarification_history: List[str] = field(
        default_factory=list
    )

    candidate_history: List[str] = field(
        default_factory=list
    )

    clarification_turns: int = 0

    retrieval_score_history: List[float] = field(
        default_factory=list
    )

    user_messages: List[str] = field(
        default_factory=list
    )

    # -----------------------------------------
    # Utility
    # -----------------------------------------

    def to_dict(self):

        return {
            key: value
            for key, value
            in self.__dict__.items()
        }
