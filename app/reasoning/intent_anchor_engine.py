from collections import Counter


INTENT_PATTERNS = {

    "Backend Engineering": [

        "backend",
        "server-side",
        "api",
        "apis",
        "database",
        "python",
        "java",
        "django",
        "flask",
        "spring"
    ],

    "Frontend Engineering": [

        "frontend",
        "ui",
        "ux",
        "react",
        "angular",
        "javascript",
        "css",
        "html"
    ],

    "DevOps Engineering": [

        "devops",
        "deployment",
        "infrastructure",
        "kubernetes",
        "docker",
        "ci/cd"
    ],

    "Data Analytics": [

        "analytics",
        "analysis",
        "data",
        "reporting",
        "insights",
        "business intelligence"
    ],

    "Leadership & Management": [

        "manager",
        "leadership",
        "executive",
        "director",
        "organizational"
    ]
}


def infer_primary_intent(
    conversation_state
):

    text_fragments = []

    # -----------------------------------------
    # Role
    # -----------------------------------------

    role = conversation_state.get(
        "role_focus"
    )

    if role:

        text_fragments.append(role)

    # -----------------------------------------
    # Technologies
    # -----------------------------------------

    technologies = conversation_state.get(
        "technology_stack",
        []
    )

    text_fragments.extend(
        technologies
    )

    # -----------------------------------------
    # Domains
    # -----------------------------------------

    domains = conversation_state.get(
        "domains",
        []
    )

    text_fragments.extend(
        domains
    )

    # -----------------------------------------
    # Competencies
    # -----------------------------------------

    competencies = conversation_state.get(
        "competencies",
        []
    )

    text_fragments.extend(
        competencies
    )

    combined_text = (
        " ".join(text_fragments)
        .lower()
    )

    scores = Counter()

    # -----------------------------------------
    # Intent matching
    # -----------------------------------------

    for (
        intent,
        keywords
    ) in INTENT_PATTERNS.items():

        for keyword in keywords:

            if keyword in combined_text:

                scores[intent] += 1

    if not scores:

        return None

    dominant_intent = (
        scores.most_common(1)[0][0]
    )

    return dominant_intent
