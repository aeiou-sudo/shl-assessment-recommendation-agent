import json
import logging
from pathlib import Path
import sys

# Project root setup
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from app.state.state_manager import (
    StateManager
)

from app.state.llm_state_interpreter import (
    interpret_user_message
)

from app.retrieval.semantic_search import (
    SemanticSearchEngine
)

from app.retrieval.query_synthesizer import (
    synthesize_retrieval_query
)

from app.reasoning.analyze_candidates_enriched import (
    load_reasoning_documents,
    analyze_candidate_ambiguity
)

from app.reasoning.llm_clarification_engine import (
    generate_clarification
)

from app.reasoning.confidence_engine import (
    evaluate_recommendation_confidence
)

from app.reasoning.query_strength_engine import (
    evaluate_query_strength
)

from app.reasoning.intent_anchor_engine import (
    infer_primary_intent
)

from app.reasoning.intent_filters import (
    filter_analysis_by_intent
)

from app.reasoning.trajectory_engine import (
    detect_trajectory_shift
)

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class ConversationAgent:

    def __init__(self):

        self.state_manager = (
            StateManager()
        )

        self.search_engine = (
            SemanticSearchEngine()
        )

        logger.info(
            "Initializing Reasoning Lookup..."
        )

        self.reasoning_lookup = (
            load_reasoning_documents()
        )

    # -------------------------------------------------
    # Dynamic Retrieval Breadth
    # -------------------------------------------------

    def determine_retrieval_depth(
        self,
        query_strength
    ):

        strength = query_strength.get(
            "strength",
            "moderate"
        )

        if strength == "weak":

            return 12

        if strength == "strong":

            return 5

        return 8

    # -------------------------------------------------
    # Dynamic Candidate Filtering
    # -------------------------------------------------

    def filter_candidates(
        self,
        retrieved
    ):

        if not retrieved:

            return []

        top_score = (
            retrieved[0]["score"]
        )

        dynamic_threshold = max(
            top_score * 0.72,
            top_score - 0.12
        )

        filtered = [

            candidate
            for candidate in retrieved
            if candidate["score"]
            >= dynamic_threshold
        ]

        return filtered

    # -------------------------------------------------
    # Retrieval Quality Evaluation
    # -------------------------------------------------

    def evaluate_retrieval_quality(
        self,
        candidates
    ):

        if not candidates:

            return {

                "quality": "failed",

                "reasoning":
                    "No candidates retrieved."
            }

        top_score = (
            candidates[0]["score"]
        )

        if top_score < 0.30:

            return {

                "quality": "weak",

                "reasoning":
                    "Top semantic similarity is weak."
            }

        if len(candidates) >= 2:

            second_score = (
                candidates[1]["score"]
            )

            gap = (
                top_score -
                second_score
            )

            if gap < 0.02:

                return {

                    "quality":
                        "ambiguous",

                    "reasoning":
                        "Top candidates are semantically clustered."
                }

        return {

            "quality":
                "usable",

            "reasoning":
                "Retrieval quality acceptable."
        }

    # -------------------------------------------------
    # Main Processing
    # -------------------------------------------------

    def process_user_message(
        self,
        user_message
    ):

        # -------------------------------------------------
        # 1. Detect Trajectory
        # -------------------------------------------------

        trajectory = (
            detect_trajectory_shift(
                self.state_manager.get_state(),
                user_message
            )
        )

        print(
            "\n--- TRAJECTORY ANALYSIS ---"
        )

        print(trajectory)

        trajectory_type = trajectory.get(
            "trajectory_type"
        )

        if trajectory_type == "pivot":

            print(
                "\n[STATE] Performing SOFT RESET"
            )

            self.state_manager.soft_reset()

        elif trajectory_type == "hard_reset":

            print(
                "\n[STATE] Performing HARD RESET"
            )

            self.state_manager.hard_reset()

        # -------------------------------------------------
        # 2. Refresh State AFTER Reset
        # -------------------------------------------------

        current_state = (
            self.state_manager.get_state()
        )

        # -------------------------------------------------
        # 3. Interpret User Message
        # -------------------------------------------------

        interpretation = (
            interpret_user_message(
                current_state,
                user_message
            )
        )

        self.state_manager.state.user_messages.append(
            user_message
        )

        if (
            interpretation.get(
                "intent_classification"
            )
            != "assessment_recommendation"
        ):

            return {

                "status":
                    "out_of_scope",

                "message": (
                    "This assistant only supports "
                    "recruitment assessment "
                    "recommendations."
                )
            }

        # -------------------------------------------------
        # 4. Apply State Operations
        # -------------------------------------------------

        self.state_manager.apply_operations(
            interpretation.get(
                "operations",
                []
            )
        )

        updated_state = (
            self.state_manager.get_state()
        )

        # -------------------------------------------------
        # 5. Infer Semantic Intent
        # -------------------------------------------------

        primary_intent = (
            infer_primary_intent(
                updated_state
            )
        )

        print(
            "\n--- PRIMARY INTENT ---"
        )

        print(primary_intent)

        if primary_intent:

            self.state_manager.state.primary_intent = (
                primary_intent
            )

            self.state_manager.state.intent_history.append(
                primary_intent
            )

        # -------------------------------------------------
        # 6. Evaluate Query Strength
        # -------------------------------------------------

        query_strength = (
            evaluate_query_strength(
                updated_state
            )
        )

        retrieval_depth = (
            self.determine_retrieval_depth(
                query_strength
            )
        )

        # -------------------------------------------------
        # 7. Weak Query Clarification
        # -------------------------------------------------

        if (
            query_strength.get(
                "strength"
            )
            == "weak"
        ):

            clarification = (
                generate_clarification(
                    user_message,
                    {
                        "query_strength":
                            query_strength
                    }
                )
            )

            return {

                "status":
                    "clarification_required",

                "query_strength":
                    query_strength,

                "clarification":
                    clarification
            }

        # -------------------------------------------------
        # 8. Synthesize Retrieval Query
        # -------------------------------------------------

        retrieval_query = (
            synthesize_retrieval_query(
                updated_state
            )
        )

        print(
            "\n--- SYNTHESIZED QUERY ---\n"
        )

        print(retrieval_query)

        if not retrieval_query:

            return {

                "status":
                    "need_input",

                "message": (
                    "Please provide more "
                    "specific requirements."
                )
            }

        # -------------------------------------------------
        # 9. Semantic Retrieval
        # -------------------------------------------------

        retrieved = (
            self.search_engine.search(
                retrieval_query,
                top_k=retrieval_depth
            )
        )

        if not retrieved:

            return {

                "status":
                    "no_results",

                "message": (
                    f"No assessments found "
                    f"for '{retrieval_query}'."
                )
            }

        # -------------------------------------------------
        # 10. Dynamic Filtering
        # -------------------------------------------------

        filtered_candidates = (
            self.filter_candidates(
                retrieved
            )
        )

        print(
            "\n--- FILTERED RETRIEVAL RESULTS ---\n"
        )

        for candidate in filtered_candidates:

            print(
                f"{candidate['name']} "
                f"| score={candidate['score']}"
            )

        # -------------------------------------------------
        # 11. Evaluate Retrieval Quality
        # -------------------------------------------------

        retrieval_quality = (
            self.evaluate_retrieval_quality(
                filtered_candidates
            )
        )

        print(
            "\n--- RETRIEVAL QUALITY ---"
        )

        print(retrieval_quality)

        # -------------------------------------------------
        # 12. Weak Retrieval
        # -------------------------------------------------

        if (
            retrieval_quality["quality"]
            == "weak"
        ):

            clarification = (
                generate_clarification(
                    retrieval_query,
                    {
                        "retrieval_quality":
                            retrieval_quality
                    }
                )
            )

            return {

                "status":
                    "clarification_required",

                "clarification":
                    clarification
            }

        # -------------------------------------------------
        # 13. Confidence Evaluation
        # -------------------------------------------------

        confidence_result = (
            evaluate_recommendation_confidence(
                filtered_candidates,
                query_strength
            )
        )

        # -------------------------------------------------
        # 14. Final Recommendation
        # -------------------------------------------------

        if (
            confidence_result["decision"]
            == "finalize"
        ):

            return {

                "status":
                    "finalized",

                "confidence":
                    confidence_result[
                        "confidence"
                    ],

                "reasoning":
                    confidence_result[
                        "reasoning"
                    ],

                "recommendation":
                    confidence_result[
                        "recommended_candidate"
                    ]
            }

        # -------------------------------------------------
        # 15. Closest Match
        # -------------------------------------------------

        if (
            confidence_result["decision"]
            == "closest_match"
        ):

            return {

                "status":
                    "closest_match",

                "confidence":
                    confidence_result[
                        "confidence"
                    ],

                "reasoning":
                    confidence_result[
                        "reasoning"
                    ],

                "recommendation":
                    confidence_result[
                        "recommended_candidate"
                    ]
            }

        # -------------------------------------------------
        # 16. Ambiguity Analysis
        # -------------------------------------------------

        analysis = (
            analyze_candidate_ambiguity(
                filtered_candidates,
                self.reasoning_lookup
            )
        )

        if primary_intent:

            analysis = (
                filter_analysis_by_intent(
                    analysis,
                    primary_intent
                )
            )

        print(
            "\n--- FILTERED ROLE DISTRIBUTION ---"
        )

        roles = analysis.get(
            "role_signal_distribution",
            {}
        )

        for role, count in roles.items():

            print(
                f"{role} | {count}"
            )

        # -------------------------------------------------
        # 17. Clarification Generation
        # -------------------------------------------------

        clarification = (
            generate_clarification(
                retrieval_query,
                analysis
            )
        )

        return {

            "status":
                "clarification_required",

            "updated_state":
                updated_state,

            "retrieval_query":
                retrieval_query,

            "retrieved_candidates":
                [
                    c["name"]
                    for c in filtered_candidates
                ],

            "clarification":
                clarification
        }


if __name__ == "__main__":

    agent = ConversationAgent()

    print(
        "\nSHL Assessment "
        "Recommendation Agent Ready"
    )

    while True:

        user_input = input(
            "\nUser: "
        )

        if (
            user_input.lower()
            in ["exit", "quit"]
        ):

            break

        result = (
            agent.process_user_message(
                user_input
            )
        )

        status = result["status"]

        if status in [
            "finalized",
            "closest_match"
        ]:

            print(
                f"\nAGENT "
                f"[{status.upper()}]: "
                f"{result['recommendation']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence']}"
            )

            print(
                f"Reasoning: "
                f"{result['reasoning']}"
            )

        elif (
            status
            == "clarification_required"
        ):

            print(
                f"\nAGENT: "
                f"{result['clarification']['clarification_question']}"
            )

        else:

            print(
                f"\nAGENT: "
                f"{result.get('message', 'Processing...')}"
            )