import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

# -----------------------------------------
# Project root setup
# -----------------------------------------

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

# -----------------------------------------
# Retrieval engine
# -----------------------------------------

from app.retrieval.semantic_search import SemanticSearchEngine

from app.reasoning.llm_clarification_engine import (
    generate_clarification
)

# -----------------------------------------
# Reasoning docs
# -----------------------------------------

REASONING_DOCS_PATH = Path(
    "generated/reasoning_documents_enriched.json"
)

# -----------------------------------------
# Hierarchy weights
# -----------------------------------------

HIERARCHY_WEIGHTS = {

    "technology_distribution": 0.7,

    "domain_distribution": 0.85,

    "competency_distribution": 0.5,

    "role_signal_distribution": 1.0,

    "seniority_distribution": 0.9,

    "assessment_intent_distribution": 0.2
}


# -----------------------------------------
# Load reasoning documents
# -----------------------------------------

def load_reasoning_documents():

    with open(
        REASONING_DOCS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        docs = json.load(f)

    # entity_id → reasoning_doc
    lookup = {}

    for doc in docs:
        lookup[doc["entity_id"]] = doc

    return lookup


# -----------------------------------------
# Aggregate ambiguity distributions
# -----------------------------------------

def analyze_candidate_ambiguity(
    retrieved_candidates,
    reasoning_lookup
):

    analysis = {

        "technology_distribution": Counter(),

        "domain_distribution": Counter(),

        "competency_distribution": Counter(),

        "role_signal_distribution": Counter(),

        "seniority_distribution": Counter(),

        "assessment_intent_distribution": Counter(),

        "candidate_names": []
    }

    # -----------------------------------------
    # Analyze each candidate
    # -----------------------------------------

    for candidate in retrieved_candidates:

        entity_id = candidate["entity_id"]

        reasoning_doc = reasoning_lookup.get(entity_id)

        if not reasoning_doc:
            continue

        analysis["candidate_names"].append(
            candidate["name"]
        )

        semantic = reasoning_doc.get(
            "semantic_reasoning",
            {}
        )

        # -----------------------------------------
        # Technologies
        # -----------------------------------------

        for item in semantic.get(
            "technologies",
            []
        ):

            analysis[
                "technology_distribution"
            ][item] += 1

        # -----------------------------------------
        # Domains
        # -----------------------------------------

        for item in semantic.get(
            "domains",
            []
        ):

            analysis[
                "domain_distribution"
            ][item] += 1

        # -----------------------------------------
        # Competencies
        # -----------------------------------------

        for item in semantic.get(
            "core_competencies",
            []
        ):

            analysis[
                "competency_distribution"
            ][item] += 1

        # -----------------------------------------
        # Role Signals
        # -----------------------------------------

        for item in semantic.get(
            "role_signals",
            []
        ):

            analysis[
                "role_signal_distribution"
            ][item] += 1

        # -----------------------------------------
        # Seniority
        # -----------------------------------------

        for item in semantic.get(
            "seniority_signals",
            []
        ):

            analysis[
                "seniority_distribution"
            ][item] += 1

        # -----------------------------------------
        # Assessment Intent
        # -----------------------------------------

        for item in semantic.get(
            "assessment_intent",
            []
        ):

            analysis[
                "assessment_intent_distribution"
            ][item] += 1

    return analysis

# -----------------------------------------
# Compute ambiguity priorities
# -----------------------------------------

def compute_ambiguity_priorities(analysis):

    priorities = []

    for dimension, weight in HIERARCHY_WEIGHTS.items():

        counter = analysis.get(dimension)

        if not counter:
            continue

        total = sum(counter.values())

        unique = len(counter)

        if total == 0:
            continue

        # -----------------------------------------
        # Diversity ambiguity score
        # -----------------------------------------

        ambiguity_score = unique / total

        # -----------------------------------------
        # Final weighted priority
        # -----------------------------------------

        priority_score = (
            ambiguity_score * weight
        )

        priorities.append({

            "dimension": dimension,

            "weight": round(weight, 3),

            "ambiguity_score": round(
                ambiguity_score,
                3
            ),

            "priority_score": round(
                priority_score,
                3
            ),

            "unique_items": unique,

            "total_items": total
        })

    # -----------------------------------------
    # Highest priority first
    # -----------------------------------------

    priorities.sort(
        key=lambda x: x["priority_score"],
        reverse=True
    )

    return priorities

# -----------------------------------------
# Pretty print
# -----------------------------------------

def print_analysis(query, analysis):

    divider = "=" * 90

    print(f"\n{divider}")
    print(f"SEMANTIC AMBIGUITY ANALYSIS")
    print(f"QUERY: {query}")
    print(divider)

    # -----------------------------------------
    # Candidates
    # -----------------------------------------

    print("\n[1] RETRIEVED CANDIDATES\n")

    for idx, name in enumerate(
        analysis["candidate_names"],
        start=1
    ):

        print(f" {idx}. {name}")

    # -----------------------------------------
    # Helper
    # -----------------------------------------

    def print_distribution(title, counter):

        print(f"\n{title}\n")

        if not counter:
            print(" No signals detected.")
            return

        for item, count in counter.most_common():

            print(f" • {item:40} | {count}")

    # -----------------------------------------
    # Print distributions
    # -----------------------------------------

    print_distribution(
        "[2] TECHNOLOGY DISTRIBUTION",
        analysis["technology_distribution"]
    )

    print_distribution(
        "[3] DOMAIN DISTRIBUTION",
        analysis["domain_distribution"]
    )

    print_distribution(
        "[4] COMPETENCY DISTRIBUTION",
        analysis["competency_distribution"]
    )

    print_distribution(
        "[5] ROLE SIGNAL DISTRIBUTION",
        analysis["role_signal_distribution"]
    )

    print_distribution(
        "[6] SENIORITY DISTRIBUTION",
        analysis["seniority_distribution"]
    )

    print_distribution(
        "[7] ASSESSMENT INTENT DISTRIBUTION",
        analysis["assessment_intent_distribution"]
    )

    print(f"\n{divider}\n")

# -----------------------------------------
# Print ambiguity priorities
# -----------------------------------------

def print_priorities(priorities):

    print("\n[8] AMBIGUITY PRIORITY RANKING\n")

    for idx, item in enumerate(
        priorities,
        start=1
    ):

        print(
            f"{idx}. "
            f"{item['dimension']}"
        )

        print(
            f"    "
            f"Hierarchy Weight : "
            f"{item['weight']}"
        )

        print(
            f"    "
            f"Ambiguity Score  : "
            f"{item['ambiguity_score']}"
        )

        print(
            f"    "
            f"Priority Score   : "
            f"{item['priority_score']}"
        )

        print(
            f"    "
            f"Unique/Total     : "
            f"{item['unique_items']}/"
            f"{item['total_items']}"
        )

        print()

# -----------------------------------------
# Main
# -----------------------------------------

if __name__ == "__main__":

    # -----------------------------------------
    # Load systems
    # -----------------------------------------

    engine = SemanticSearchEngine()

    reasoning_lookup = load_reasoning_documents()

    # -----------------------------------------
    # Query
    # -----------------------------------------

    query = "Assessment for backend python developer"

    retrieved = engine.search(
        query,
        top_k=10
    )

    # -----------------------------------------
    # Analyze ambiguity
    # -----------------------------------------

    analysis = analyze_candidate_ambiguity(
        retrieved,
        reasoning_lookup
    )

    # -----------------------------------------
    # Compute priorities
    # -----------------------------------------

    priorities = compute_ambiguity_priorities(
        analysis
    )

    # -----------------------------------------
    # Print
    # -----------------------------------------

    print_analysis(query, analysis)

    print_priorities(priorities)

    # -----------------------------------------
    # LLM clarification reasoning
    # -----------------------------------------

    clarification = generate_clarification(
        query,
        analysis
    )

    print("\n[9] LLM CLARIFICATION REASONING\n")

    print(
        "Dominant Ambiguity:\n"
    )

    print(
        clarification["dominant_ambiguity"]
    )

    print("\nReasoning:\n")

    print(
        clarification["reasoning"]
    )

    print("\nClarification Question:\n")

    print(
        clarification["clarification_question"]
    )