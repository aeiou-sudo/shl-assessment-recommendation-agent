def evaluate_query_strength(
    conversation_state
):
    """
    Evaluates how semantically complete
    the current user specification is.
    """

    score = 0

    matched_dimensions = []

    # -----------------------------------------
    # Role focus
    # -----------------------------------------

    if conversation_state.get(
        "role_focus"
    ):

        score += 2

        matched_dimensions.append(
            "role_focus"
        )

    # -----------------------------------------
    # Technology stack
    # -----------------------------------------

    technologies = conversation_state.get(
        "technology_stack",
        []
    )

    if technologies:

        score += 2

        matched_dimensions.append(
            "technology_stack"
        )

    # -----------------------------------------
    # Seniority
    # -----------------------------------------

    if conversation_state.get(
        "seniority"
    ):

        score += 2

        matched_dimensions.append(
            "seniority"
        )

    # -----------------------------------------
    # Competencies
    # -----------------------------------------

    competencies = conversation_state.get(
        "competencies",
        []
    )

    if competencies:

        score += 1

        matched_dimensions.append(
            "competencies"
        )

    # -----------------------------------------
    # Domains
    # -----------------------------------------

    domains = conversation_state.get(
        "domains",
        []
    )

    if domains:

        score += 1

        matched_dimensions.append(
            "domains"
        )

    # -----------------------------------------
    # Role modifiers
    # -----------------------------------------

    if (
        conversation_state.get(
            "include_devops"
        )
        is not None
    ):

        score += 1

        matched_dimensions.append(
            "include_devops"
        )

    if (
        conversation_state.get(
            "include_frontend"
        )
        is not None
    ):

        score += 1

        matched_dimensions.append(
            "include_frontend"
        )

    # -----------------------------------------
    # Strength classification
    # -----------------------------------------

    if score <= 2:

        strength = "weak"

    elif score <= 5:

        strength = "moderate"

    else:

        strength = "strong"

    return {

        "score": score,

        "strength": strength,

        "matched_dimensions":
            matched_dimensions
    }
