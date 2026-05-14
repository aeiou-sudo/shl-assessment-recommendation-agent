INTENT_ALLOWED_DOMAINS = {

    "Backend Engineering": [

        "Backend Developer",
        "Software Development",
        "API Development",
        "Server-side Development",
        "DevOps Engineering"
    ],

    "Frontend Engineering": [

        "Frontend Development",
        "UI Development",
        "Web Development"
    ],

    "Data Analytics": [

        "Data Analysis",
        "Business Intelligence",
        "Analytics"
    ]
}


def filter_analysis_by_intent(
    analysis,
    primary_intent
):

    if (
        primary_intent
        not in INTENT_ALLOWED_DOMAINS
    ):

        return analysis

    allowed = (
        INTENT_ALLOWED_DOMAINS[
            primary_intent
        ]
    )

    role_distribution = analysis.get(
        "role_signal_distribution",
        {}
    )

    filtered_roles = {

        role: count

        for role, count
        in role_distribution.items()

        if any(
            allowed_term.lower()
            in role.lower()

            for allowed_term
            in allowed
        )
    }

    analysis[
        "role_signal_distribution"
    ] = filtered_roles

    return analysis
