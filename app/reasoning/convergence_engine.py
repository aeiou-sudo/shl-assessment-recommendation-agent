from collections import Counter


def has_semantic_convergence(
    conversation_state,
    query_strength,
    clarification_turns_threshold=3
):
    """
    Determines whether the conversation
    has semantically stabilized enough
    to finalize recommendation.
    """

    # -----------------------------------------
    # Query maturity required
    # -----------------------------------------

    if (
        query_strength["strength"]
        == "weak"
    ):

        return False

    # -----------------------------------------
    # Enough clarification turns
    # -----------------------------------------

    clarification_turns = (
        conversation_state.get(
            "clarification_turns",
            0
        )
    )

    if (
        clarification_turns
        < clarification_turns_threshold
    ):

        return False

    # -----------------------------------------
    # Intent stabilization
    # -----------------------------------------

    intent_history = (
        conversation_state.get(
            "intent_history",
            []
        )
    )

    if len(intent_history) < 3:

        return False

    recent_intents = (
        intent_history[-3:]
    )

    if len(set(recent_intents)) != 1:

        return False

    # -----------------------------------------
    # Candidate stabilization
    # -----------------------------------------

    candidate_history = (
        conversation_state.get(
            "candidate_history",
            []
        )
    )

    if len(candidate_history) < 3:

        return False

    recent_candidates = (
        candidate_history[-3:]
    )

    if len(set(recent_candidates)) != 1:

        return False

    return True
