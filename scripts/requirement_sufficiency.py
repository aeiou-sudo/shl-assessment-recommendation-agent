import json


with open("data/requirement_dimensions.json", "r") as f:
    REQUIREMENT_DIMENSIONS = json.load(f)


def evaluate_requirement_sufficiency(conversation_state):
    """
    Determines whether enough recruiter information
    exists to generate reliable assessment recommendations.
    """

    critical_dimensions = REQUIREMENT_DIMENSIONS["critical_dimensions"]
    important_dimensions = REQUIREMENT_DIMENSIONS["important_dimensions"]

    missing_critical = []
    missing_important = []

    # ----------------------------------------
    # CHECK CRITICAL DIMENSIONS
    # ----------------------------------------

    if not conversation_state.get("role_titles"):
        missing_critical.append("role_title")

    technical_skills = (
        conversation_state
        .get("semantic_requirements", {})
        .get("technical_skills", [])
    )

    if not technical_skills:
        missing_critical.append("technical_skills")

    job_levels = (
        conversation_state
        .get("explicit_constraints", {})
        .get("job_levels", [])
    )

    if not job_levels:
        missing_critical.append("experience_level")

    # ----------------------------------------
    # CHECK IMPORTANT DIMENSIONS
    # ----------------------------------------

    competencies = (
        conversation_state
        .get("semantic_requirements", {})
        .get("competencies", [])
    )

    behavioral_traits = (
        conversation_state
        .get("semantic_requirements", {})
        .get("behavioral_traits", [])
    )

    domains = (
        conversation_state
        .get("semantic_requirements", {})
        .get("domains", [])
    )

    if not competencies:
        missing_important.append("problem_solving")

    if not behavioral_traits:
        missing_important.append("behavioral_evaluation")

    if not domains:
        missing_important.append("specialization")

    # ----------------------------------------
    # COMPLETENESS SCORING
    # ----------------------------------------

    total_critical = len(critical_dimensions)
    total_important = len(important_dimensions)

    critical_score = (
        total_critical - len(missing_critical)
    ) / total_critical

    important_score = (
        total_important - len(missing_important)
    ) / total_important

    # weighted score
    completeness_score = (
        critical_score * 0.7
        + important_score * 0.3
    )

    # ----------------------------------------
    # RECOMMENDATION READINESS
    # ----------------------------------------

    ready_for_recommendation = (
        len(missing_critical) == 0
        and completeness_score >= 0.65
    )

    return {
        "intent_completeness_score": round(completeness_score, 2),
        "ready_for_recommendation": ready_for_recommendation,
        "missing_critical_dimensions": missing_critical,
        "missing_important_dimensions": missing_important
    }
