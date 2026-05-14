def evaluate_recommendation_confidence(
    candidates,
    query_strength
):
    """
    Evaluates whether the current retrieval
    results are strong enough to:
    
    - finalize recommendation
    - request clarification
    - fallback to closest match

    This engine uses:
    - query maturity
    - top score quality
    - score separation
    - retrieval dominance
    """

    # -----------------------------------------
    # No candidates
    # -----------------------------------------

    if not candidates:

        return {

            "decision": "clarify",

            "confidence": 0.0,

            "reasoning":
                "No relevant candidates found.",

            "recommended_candidate": None
        }

    # -----------------------------------------
    # Query maturity
    # -----------------------------------------

    strength = query_strength["strength"]

    # -----------------------------------------
    # Top candidate
    # -----------------------------------------

    top_candidate = candidates[0]

    top_score = top_candidate["score"]

    # -----------------------------------------
    # Second candidate
    # -----------------------------------------

    if len(candidates) > 1:

        second_score = candidates[1]["score"]

    else:

        second_score = 0.0

    # -----------------------------------------
    # Dominance gap
    # -----------------------------------------

    score_gap = top_score - second_score

    # -----------------------------------------
    # Relative dominance ratio
    # -----------------------------------------

    if second_score > 0:

        dominance_ratio = (
            top_score / second_score
        )

    else:

        dominance_ratio = 999.0

    # -----------------------------------------
    # DEBUG
    # -----------------------------------------

    print("\n--- CONFIDENCE ANALYSIS ---\n")

    print(f"Query Strength  : {strength}")

    print(f"Top Score       : {top_score:.4f}")

    print(f"Second Score    : {second_score:.4f}")

    print(f"Score Gap       : {score_gap:.4f}")

    print(
        f"Dominance Ratio : "
        f"{dominance_ratio:.4f}"
    )

    # =================================================
    # CASE 1 — Weak Query
    # =================================================

    if strength == "weak":

        return {

            "decision": "clarify",

            "confidence": top_score,

            "reasoning": (
                "Query is too weak for "
                "confident recommendation."
            ),

            "recommended_candidate": None
        }

    # =================================================
    # CASE 2 — Strong Dominant Recommendation
    # =================================================

    strong_dominance = (

        top_score >= 0.55

        and

        (
            score_gap >= 0.15
            or
            dominance_ratio >= 1.5
        )
    )

    if strong_dominance:

        return {

            "decision": "finalize",

            "confidence": round(top_score, 4),

            "reasoning": (
                "Strong recommendation "
                "dominance detected."
            ),

            "recommended_candidate":
                top_candidate
        }

    # =================================================
    # CASE 3 — Ambiguous Competition
    # =================================================

    ambiguous_competition = (

        abs(score_gap) <= 0.08
    )

    if ambiguous_competition:

        return {

            "decision": "clarify",

            "confidence": round(top_score, 4),

            "reasoning": (
                "Multiple competing "
                "recommendations detected."
            ),

            "recommended_candidate": None
        }

    # =================================================
    # CASE 4 — Weak Retrieval Quality
    # =================================================

    weak_retrieval = (

        top_score < 0.50
    )

    if weak_retrieval:

        if strength == "strong":

            return {

                "decision": "closest_match",

                "confidence": round(top_score, 4),

                "reasoning": (
                    "No highly confident "
                    "recommendation found. "
                    "Returning closest match."
                ),

                "recommended_candidate":
                    top_candidate
            }

        return {

            "decision": "clarify",

            "confidence": round(top_score, 4),

            "reasoning": (
                "Retrieval confidence "
                "is currently weak."
            ),

            "recommended_candidate": None
        }

    # =================================================
    # CASE 5 — Moderate Recommendation
    # =================================================

    moderate_confidence = (

        top_score >= 0.50

        and

        dominance_ratio >= 1.25
    )

    if moderate_confidence:

        return {

            "decision": "finalize",

            "confidence": round(top_score, 4),

            "reasoning": (
                "Moderately dominant "
                "recommendation identified."
            ),

            "recommended_candidate":
                top_candidate
        }

    # =================================================
    # DEFAULT — Clarify
    # =================================================

    return {

        "decision": "clarify",

        "confidence": round(top_score, 4),

        "reasoning": (
            "Additional clarification "
            "is required."
        ),

        "recommended_candidate": None
    }
