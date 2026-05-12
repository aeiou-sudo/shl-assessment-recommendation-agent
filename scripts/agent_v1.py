import os
import json
import copy
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================
# CONFIG
# ============================================================

SEARCH_DOCUMENTS_PATH = "data/processed/search_documents.json"
TOP_K = 5

# ============================================================
# LOAD DATA
# ============================================================

with open(SEARCH_DOCUMENTS_PATH, "r") as f:
    documents = json.load(f)

def normalize_document(doc):

    return {

        "name":
            doc.get("name", ""),

        "description":
            doc.get("description", ""),

        "url":
            doc.get("url")
            or doc.get("link", ""),

        "job_levels":
            doc.get("job_levels", []),

        "languages":
            doc.get("languages", []),

        "duration":
            doc.get("duration", ""),

        "remote":
            doc.get("remote", ""),

        "adaptive":
            doc.get("adaptive", ""),

        "keys":
            doc.get("keys", []),

        "test_type":
            doc.get("test_type", "")
    }

documents = [
    normalize_document(doc)
    for doc in documents
]

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Using device: mps")
print("Loading upgraded 384-dim model (snowflake/snowflake-arctic-embed-s)...")

model = SentenceTransformer(
    "snowflake/snowflake-arctic-embed-s"
)

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("Generating document embeddings...")

document_texts = []

for doc in documents:

    searchable_text = " ".join([

        doc.get("name", ""),

        doc.get("description", ""),

        " ".join(doc.get("job_levels", [])),

        " ".join(doc.get("languages", [])),

        " ".join(doc.get("keys", []))
    ])

    document_texts.append(searchable_text)

embedding_matrix = model.encode(
    document_texts,
    convert_to_numpy=True,
    show_progress_bar=True
).astype("float32")

# ============================================================
# BUILD FAISS INDEX
# ============================================================

index = faiss.IndexFlatL2(
    embedding_matrix.shape[1]
)

index.add(embedding_matrix)

print("\nLoading retrieval system...")
print("\nSemantic retrieval system ready.\n")

# ============================================================
# EMPTY STATE
# ============================================================

EMPTY_STATE = {
    "role_titles": [],
    "explicit_constraints": {
        "job_levels": [],
        "remote": None,
        "adaptive": None,
        "duration": None,
        "languages": []
    },
    "semantic_requirements": {
        "technical_skills": [],
        "domains": [],
        "competencies": [],
        "behavioral_traits": [],
        "engineering_focus": []
    }
}

conversation_state = copy.deepcopy(
    EMPTY_STATE
)

# ============================================================
# TAXONOMIES
# ============================================================

ROLE_KEYWORDS = {

    "front end developer": "Front End Developer",
    "frontend developer": "Front End Developer",

    "back end developer": "Backend Developer",
    "backend developer": "Backend Developer",

    "java backend engineer": "Java Backend Engineer",

    "data scientist": "Data Scientist",

    "machine learning engineer": "Machine Learning Engineer",

    "leadership": "Leadership",

    "manager": "Manager",

    "executive": "Executive"
}

JOB_LEVEL_KEYWORDS = {

    "entry level": "Entry-Level",

    "fresher": "Entry-Level",

    "beginner": "Entry-Level",

    "graduate": "Graduate",

    "mid level": "Mid-Professional",

    "senior": "Mid-Professional",

    "manager": "Manager",

    "director": "Director",

    "executive": "Executive"
}

TECHNICAL_SKILLS = [

    "html",
    "css",
    "javascript",
    "java",
    "python",
    "sql",
    "react",
    "nodejs",
    "node.js",
    "rest api",
    "restful api",
    "spring boot",
    "machine learning",
    "deep learning"
]

BEHAVIORAL_TRAITS = [

    "communication",

    "analytical",

    "analytical skills",

    "leadership",

    "problem solving",

    "teamwork",

    "adaptability",

    "critical thinking"
]

DOMAIN_KEYWORDS = {

    "frontend": "frontend development",

    "front end": "frontend development",

    "backend": "backend development",

    "api": "api development",

    "web services": "web services",

    "distributed systems": "distributed systems"
}

ENGINEERING_FOCUS = {

    "application development": "application development",

    "distributed systems": "distributed systems",

    "system design": "system design"
}

LANGUAGE_KEYWORDS = [

    "english",

    "spanish",

    "french"
]

# ============================================================
# UTILITIES
# ============================================================

def normalize_text(text):

    return text.lower().strip()

def unique_extend(target_list, values):

    for v in values:

        if v not in target_list:

            target_list.append(v)

# ============================================================
# INTENT EXTRACTION
# ============================================================

def extract_intent(query):

    query_lower = normalize_text(query)

    extracted = copy.deepcopy(
        EMPTY_STATE
    )

    # ========================================================
    # ROLE TITLES
    # ========================================================

    for keyword, role in ROLE_KEYWORDS.items():

        if keyword in query_lower:

            extracted["role_titles"].append(role)

    # ========================================================
    # JOB LEVELS
    # ========================================================

    for keyword, level in JOB_LEVEL_KEYWORDS.items():

        if keyword in query_lower:

            extracted["explicit_constraints"][
                "job_levels"
            ].append(level)

    # ========================================================
    # TECHNICAL SKILLS
    # ========================================================

    for skill in TECHNICAL_SKILLS:

        if skill in query_lower:

            extracted["semantic_requirements"][
                "technical_skills"
            ].append(skill)

    # ========================================================
    # DOMAINS
    # ========================================================

    for keyword, domain in DOMAIN_KEYWORDS.items():

        if keyword in query_lower:

            extracted["semantic_requirements"][
                "domains"
            ].append(domain)

    # ========================================================
    # BEHAVIORAL + COMMUNICATION NORMALIZATION
    # ========================================================

    if "communication" in query_lower:

        extracted["semantic_requirements"][
            "behavioral_traits"
        ].append("communication")

        extracted["semantic_requirements"][
            "competencies"
        ].append("business communication")

    for trait in BEHAVIORAL_TRAITS:

        if trait == "communication":
            continue

        if trait in query_lower:

            extracted["semantic_requirements"][
                "behavioral_traits"
            ].append(trait)

    # ========================================================
    # ENGINEERING FOCUS
    # ========================================================

    for keyword, focus in ENGINEERING_FOCUS.items():

        if keyword in query_lower:

            extracted["semantic_requirements"][
                "engineering_focus"
            ].append(focus)

    # ========================================================
    # LANGUAGES
    # ========================================================

    for lang in LANGUAGE_KEYWORDS:

        if lang in query_lower:

            extracted["explicit_constraints"][
                "languages"
            ].append(lang)

    return extracted

# ============================================================
# STATE MUTATION
# ============================================================

def merge_state(existing_state, extracted_state):

    merged = copy.deepcopy(existing_state)

    # ========================================================
    # ROLE TITLES
    # ========================================================

    unique_extend(
        merged["role_titles"],
        extracted_state["role_titles"]
    )

    # ========================================================
    # JOB LEVELS
    # ========================================================

    unique_extend(
        merged["explicit_constraints"]["job_levels"],
        extracted_state["explicit_constraints"]["job_levels"]
    )

    # ========================================================
    # LANGUAGES
    # ========================================================

    unique_extend(
        merged["explicit_constraints"]["languages"],
        extracted_state["explicit_constraints"]["languages"]
    )

    # ========================================================
    # TECHNICAL SKILLS
    # ========================================================

    unique_extend(
        merged["semantic_requirements"]["technical_skills"],
        extracted_state["semantic_requirements"]["technical_skills"]
    )

    # ========================================================
    # DOMAINS
    # ========================================================

    unique_extend(
        merged["semantic_requirements"]["domains"],
        extracted_state["semantic_requirements"]["domains"]
    )

    # ========================================================
    # COMPETENCIES
    # ========================================================

    unique_extend(
        merged["semantic_requirements"]["competencies"],
        extracted_state["semantic_requirements"]["competencies"]
    )

    # ========================================================
    # BEHAVIORAL TRAITS
    # ========================================================

    unique_extend(
        merged["semantic_requirements"]["behavioral_traits"],
        extracted_state["semantic_requirements"]["behavioral_traits"]
    )

    # ========================================================
    # ENGINEERING FOCUS
    # ========================================================

    unique_extend(
        merged["semantic_requirements"]["engineering_focus"],
        extracted_state["semantic_requirements"]["engineering_focus"]
    )

    return merged

# ============================================================
# REQUIREMENT SUFFICIENCY
# ============================================================

def evaluate_sufficiency(state):

    missing = []

    clarification_questions = []

    # ========================================================
    # COUNTS
    # ========================================================

    role_count = len(
        state["role_titles"]
    )

    tech_count = len(
        state["semantic_requirements"][
            "technical_skills"
        ]
    )

    behavioral_count = len(
        state["semantic_requirements"][
            "behavioral_traits"
        ]
    )

    job_level_count = len(
        state["explicit_constraints"][
            "job_levels"
        ]
    )

    # ========================================================
    # ROLE CHECK
    # ========================================================

    if role_count == 0:

        missing.append("role")

        clarification_questions.append(
            "Which role are you hiring for?"
        )

    # ========================================================
    # TECHNICAL SKILL CHECK
    # ========================================================

    if tech_count < 2:

        missing.append(
            "technical_skills"
        )

        clarification_questions.append(
            "Which technical skills should the assessment evaluate?"
        )

    # ========================================================
    # JOB LEVEL CHECK
    # ========================================================

    if job_level_count == 0:

        missing.append("job_level")

        clarification_questions.append(
            "Is this assessment intended for entry-level, mid-level, or senior candidates?"
        )

    # ========================================================
    # OPTIONAL BEHAVIORAL ENRICHMENT
    # ========================================================

    if behavioral_count == 0:

        clarification_questions.append(
            "Do you also want communication, analytical, or behavioral skill evaluation?"
        )

    # ========================================================
    # READY LOGIC
    # ========================================================

    ready = True

    if role_count == 0:
        ready = False

    if tech_count == 0 and behavioral_count == 0:
        ready = False

    if job_level_count == 0:
        ready = False

    return {
        "ready": ready,
        "missing": missing,
        "clarification_questions": clarification_questions
    }

# ============================================================
# BUILD SEARCH QUERY
# ============================================================

def build_search_query(state):

    parts = []

    parts.extend(
        state["role_titles"]
    )

    parts.extend(
        state["explicit_constraints"][
            "job_levels"
        ]
    )

    parts.extend(
        state["semantic_requirements"][
            "technical_skills"
        ]
    )

    parts.extend(
        state["semantic_requirements"][
            "domains"
        ]
    )

    parts.extend(
        state["semantic_requirements"][
            "competencies"
        ]
    )

    parts.extend(
        state["semantic_requirements"][
            "behavioral_traits"
        ]
    )

    parts.extend(
        state["semantic_requirements"][
            "engineering_focus"
        ]
    )

    cleaned_parts = [
        p for p in parts if p
    ]

    return " ".join(
        cleaned_parts
    ).strip()
# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_candidates(
    query,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [query]
    )[0].astype("float32")

    distances, indices = index.search(
        np.array([query_embedding]),
        top_k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        candidate = copy.deepcopy(
            documents[idx]
        )

        candidate["distance_score"] = float(
            distance
        )

        results.append(candidate)

    return results

# ============================================================
# EXACT MATCH SCORING
# ============================================================

def compute_exact_match_score(
    candidate,
    state
):

    score = 0

    exact_skill_matches = []

    exact_domain_matches = []

    competency_matches = []

    candidate_text = (
        candidate.get("name", "") + " " +
        candidate.get("description", "")
    ).lower()

    # ========================================================
    # TECHNICAL SKILLS
    # ========================================================

    for skill in state[
        "semantic_requirements"
    ]["technical_skills"]:

        if skill.lower() in candidate_text:

            score += 5

            exact_skill_matches.append(
                skill
            )

    # ========================================================
    # DOMAINS
    # ========================================================

    for domain in state[
        "semantic_requirements"
    ]["domains"]:

        if domain.lower() in candidate_text:

            score += 3

            exact_domain_matches.append(
                domain
            )

    # ========================================================
    # COMPETENCIES
    # ========================================================

    for competency in state[
        "semantic_requirements"
    ]["competencies"]:

        if competency.lower() in candidate_text:

            score += 2

            competency_matches.append(
                competency
            )

    return {

        "score": score,

        "exact_skill_matches":
            exact_skill_matches,

        "exact_domain_matches":
            exact_domain_matches,

        "competency_matches":
            competency_matches
    }

# ============================================================
# COMPATIBILITY SCORING
# ============================================================

def compute_compatibility(
    candidate,
    state
):

    score = 0

    notes = []

    # ========================================================
    # JOB LEVEL COMPATIBILITY
    # ========================================================

    requested_levels = state[
        "explicit_constraints"
    ]["job_levels"]

    candidate_levels = candidate.get(
        "job_levels",
        []
    )

    if requested_levels:

        overlap = set(
            requested_levels
        ).intersection(
            set(candidate_levels)
        )

        if overlap:

            score += 5

        else:

            score -= 2

            notes.append(
                "job level mismatch"
            )

    # ========================================================
    # LANGUAGE COMPATIBILITY
    # ========================================================

    requested_languages = state[
        "explicit_constraints"
    ]["languages"]

    candidate_languages = [
        x.lower()
        for x in candidate.get(
            "languages",
            []
        )
    ]

    if requested_languages:

        language_overlap = False

        for lang in requested_languages:

            if lang.lower() in " ".join(
                candidate_languages
            ):

                language_overlap = True

        if language_overlap:

            score += 2

        else:

            score -= 1

            notes.append(
                "language mismatch"
            )

    return score, notes

# ============================================================
# FINAL RESULT SCORING
# ============================================================

def score_results(
    results,
    state
):

    ranked = []

    for result in results:

        exact_match = (
            compute_exact_match_score(
                result,
                state
            )
        )

        compatibility_score, validation_notes = (
            compute_compatibility(
                result,
                state
            )
        )

        final_score = (

            exact_match["score"] * 3

            +

            compatibility_score * 2

            -

            result["distance_score"]
        )

        result["exact_match"] = (
            exact_match
        )

        result["compatibility_score"] = (
            compatibility_score
        )

        result["validation_notes"] = (
            validation_notes
        )

        result["final_score"] = (
            final_score
        )

        ranked.append(result)

    ranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return ranked

# ============================================================
# ASSESSMENT PACKAGE GENERATION
# ============================================================

def generate_assessment_package(
    state,
    ranked_results
):

    print(
        "\n=== ASSESSMENT RECOMMENDATION ===\n"
    )

    # ========================================================
    # ROLE
    # ========================================================

    roles = ", ".join(
        state["role_titles"]
    )

    print(
        f"Recommended assessment package for: {roles}"
    )

    # ========================================================
    # TECHNICAL
    # ========================================================

    tech = ", ".join(
        state["semantic_requirements"][
            "technical_skills"
        ]
    )

    if tech:

        print(
            f"\nTechnical focus: {tech}"
        )

    # ========================================================
    # BEHAVIORAL
    # ========================================================

    behavioral = ", ".join(
        state["semantic_requirements"][
            "behavioral_traits"
        ]
    )

    if behavioral:

        print(
            f"Behavioral focus: {behavioral}"
        )

    # ========================================================
    # DOMAINS
    # ========================================================

    domains = ", ".join(
        state["semantic_requirements"][
            "domains"
        ]
    )

    if domains:

        print(
            f"Engineering domains: {domains}"
        )

    # ========================================================
    # PACKAGE TABLE
    # ========================================================

    print(
        "\nRecommended Assessments:\n"
    )

    for i, item in enumerate(
        ranked_results[:5],
        start=1
    ):

        print(f"{i}. {item['name']}")

        print(
            f"   URL: {item['link']}"
        )

        print(
            f"   Duration: {item.get('duration', 'N/A')}"
        )

        print(
            f"   Job Levels: {item.get('job_levels', [])}"
        )

        print(
            f"   Languages: {item.get('languages', [])}"
        )

        print(
            f"   Remote Testing: {item.get('remote', 'N/A')}"
        )

        print(
            f"   Adaptive: {item.get('adaptive', 'N/A')}"
        )

        print(
            f"   Match Score: {item['final_score']:.2f}"
        )

        print()

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    recruiter_query = input(
        "Recruiter Query: "
    )

    if recruiter_query.lower() in [
        "exit",
        "quit"
    ]:
        break

    print(
        "\nExtracting intent...\n"
    )

    extracted = extract_intent(
        recruiter_query
    )

    conversation_state = merge_state(
        conversation_state,
        extracted
    )

    # ========================================================
    # SHOW STATE
    # ========================================================

    print(
        "=== CONVERSATION STATE ===\n"
    )

    print(
        json.dumps(
            conversation_state,
            indent=2
        )
    )

    # ========================================================
    # REQUIREMENT SUFFICIENCY
    # ========================================================

    sufficiency = evaluate_sufficiency(
        conversation_state
    )

    print(
        "\n=== REQUIREMENT ANALYSIS ===\n"
    )

    print(
        json.dumps(
            sufficiency,
            indent=2
        )
    )

    # ========================================================
    # ASK FOLLOW-UP QUESTIONS
    # ========================================================

    if not sufficiency["ready"]:

        print(
            "\n=== CLARIFICATION QUESTIONS ===\n"
        )

        for question in sufficiency[
            "clarification_questions"
        ]:

            print(f"- {question}")

        print()

        continue

    # ========================================================
    # BUILD QUERY
    # ========================================================

    search_query = build_search_query(
        conversation_state
    )

    print(
        "\n=== SEARCH QUERY ===\n"
    )

    print(search_query)

    # ========================================================
    # RETRIEVE
    # ========================================================

    results = retrieve_candidates(
        search_query
    )

    print(
        f"\nDEBUG: Found {len(results)} matches.\n"
    )

    # ========================================================
    # SCORE
    # ========================================================

    ranked_results = score_results(
        results,
        conversation_state
    )

    # ========================================================
    # TOP MATCHES
    # ========================================================

    print(
        "\n=== TOP MATCHES ===\n"
    )

    for i, item in enumerate(
        ranked_results[:5],
        start=1
    ):

        print(f"\n#{i}")

        print(
            f"Name: {item['name']}"
        )

        print(
            f"Distance Score: {item['distance_score']:.4f}"
        )

        print(
            f"URL: {item['link']}"
        )

        metadata = {

            "job_levels":
                item.get(
                    "job_levels",
                    []
                ),

            "languages":
                item.get(
                    "languages",
                    []
                ),

            "duration":
                item.get(
                    "duration",
                    ""
                ),

            "remote":
                item.get(
                    "remote",
                    ""
                ),

            "adaptive":
                item.get(
                    "adaptive",
                    ""
                ),

            "keys":
                item.get(
                    "keys",
                    []
                )
        }

        print(
            f"Metadata: {metadata}"
        )

        print(
            f"Compatibility Score: "
            f"{item['compatibility_score']}"
        )

        print(
            f"Final Score: "
            f"{item['final_score']:.4f}"
        )

        print(
            f"Validation Notes: "
            f"{item['validation_notes']}"
        )

        print(
            f"Exact Match: "
            f"{item['exact_match']}"
        )

    # ========================================================
    # FINAL PACKAGE
    # ========================================================

    generate_assessment_package(
        conversation_state,
        ranked_results
    )

    print(
        "\n" + "=" * 60 + "\n"
    )
