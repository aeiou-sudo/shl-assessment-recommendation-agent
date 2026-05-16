"""Run a single local convergence turn for quick verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.assessment_agent import AssessmentRecommendationAgent


def main() -> None:
    query = " ".join(sys.argv[1:]) or "I need an assessment for a .NET backend developer"
    agent = AssessmentRecommendationAgent.from_catalogue()
    response = agent.handle(query)
    print(
        json.dumps(
            {
                "message": response.message,
                "recommendations": response.recommendations or None,
                "assessment_plan": response.assessment_plan,
                "end_of_conversation": response.end_of_conversation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
