CLARIFICATION_QUESTION_BANK = {
    "technical_skills": [
        "Which core technologies or programming languages should the assessment focus on?"
    ],

    "experience_level": [
        "Is this role entry-level, mid-level, or senior?"
    ],

    "problem_solving": [
        "Should the assessment evaluate analytical or problem-solving ability?"
    ],

    "behavioral_evaluation": [
        "Would you also like to assess communication, collaboration, or other behavioral traits?"
    ],

    "specialization": [
        "Is there a specific engineering specialization such as backend systems, frontend development, cloud infrastructure, or data engineering?"
    ],

    "role_title": [
        "What specific role are you hiring for?"
    ]
}


def generate_clarification_questions(
    sufficiency_result,
    conversation_state
):
    """
    Prioritize clarification questions based on:
    - missing critical dimensions first
    - then important dimensions
    - avoid redundant questions
    """

    questions = []

    missing_critical = sufficiency_result.get(
        "missing_critical_dimensions",
        []
    )

    missing_important = sufficiency_result.get(
        "missing_important_dimensions",
        []
    )

    # ----------------------------------------
    # PRIORITY 1 — CRITICAL
    # ----------------------------------------

    for dimension in missing_critical:
        if dimension in CLARIFICATION_QUESTION_BANK:
            questions.extend(
                CLARIFICATION_QUESTION_BANK[dimension]
            )

    # ----------------------------------------
    # PRIORITY 2 — IMPORTANT
    # ----------------------------------------

    for dimension in missing_important:
        if dimension in CLARIFICATION_QUESTION_BANK:
            questions.extend(
                CLARIFICATION_QUESTION_BANK[dimension]
            )

    # ----------------------------------------
    # LIMIT QUESTION COUNT
    # ----------------------------------------

    # prevent overwhelming recruiter
    return questions[:2]
