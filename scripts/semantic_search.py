import json
import pickle
import os
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

import requests
from huggingface_hub import InferenceClient

# Enter this in the terminal and when prompted enter the API key
# huggingface-cli login 


import torch # Add this at the top

# 1. Choose the device (MPS is for Mac Silicon)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the upgraded 384-dim model
# snowflake-arctic-embed-s is state-of-the-art for its size
MODEL_ID = 'snowflake/snowflake-arctic-embed-s'
print(f"Loading upgraded 384-dim model ({MODEL_ID})...")

model = SentenceTransformer(MODEL_ID, device=device)

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "BAAI/bge-m3"

FAISS_INDEX_PATH = Path(
    "data/processed/faiss_index.bin"
)

METADATA_PATH = Path(
    "data/processed/vector_metadata.pkl"
)

DIMENSIONS_PATH = Path(
    "data/processed/semantic_dimensions.json"
)


# token = "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"
# if not token:
#     print("ERROR: HF_TOKEN not found in environment variables.")

# client_hf = InferenceClient(token=token)
# MODEL_ID = "BAAI/bge-m3"


conversation_state = {
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

with open(
    "data/role_expansion_map.json",
    "r"
) as f:

    ROLE_EXPANSION_MAP = json.load(f)

with open(
    "data/semantic_normalization_map.json",
    "r"
) as f:

    NORMALIZATION_MAP = json.load(f)

def is_empty_state(state):

    if state["role_titles"]:
        return False

    explicit = state[
        "explicit_constraints"
    ]

    semantic = state[
        "semantic_requirements"
    ]

    for value in explicit.values():

        if value not in [None, []]:
            return False

    for values in semantic.values():

        if values:
            return False

    return True

def load_index():

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def load_dimensions():

    with open(DIMENSIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dimension_context(dimensions):

    aggregated = dimensions["aggregated_dimensions"]

    context = []

    for category, values in aggregated.items():

        sample_values = values[:30]

        context.append(
            f"{category}: {', '.join(sample_values)}"
        )

    return "\n".join(context)

import re

def extract_intent(user_query, dimension_context):
    prompt = f""" Return ONLY valid JSON. No preamble, no markdown blocks.

AVAILABLE SEMANTIC DIMENSIONS:
{dimension_context}

### STRICT EXTRACTION & CATEGORY RULES:

1. **role_titles**: Extract specific job roles only (e.g., "Java Developer"). 
   - DO NOT include technologies, domains, or seniority here.
   role_titles must represent actual hiring job roles.

    NEVER place:
    - soft skills
    - communication traits
    - competencies
    - technologies
    - engineering domains

    Examples of INVALID role_titles:
    - communication
    - leadership
    - teamwork
    - Java
    - cloud infrastructure

2. **job_levels (STRICT EMPTY RULE)**:
   - ONLY include if a specific seniority keyword is used (e.g., "Senior", "Junior", "Lead", "Grad", "Entry", "Manager").
   - If no seniority is mentioned, return an empty list []. 
   - Mapping: Use standard tiers (Entry-Level, Mid-Professional, Professional Individual Contributor, Manager, Executive).

3. **technical_skills**: Specific technologies, frameworks, or tools explicitly mentioned.
   - DO NOT substitute or infer related tech (e.g., if "Java" is mentioned, do not add "Kotlin").
   - DO NOT include spoken languages here.

4. **languages (STRICT EMPTY RULE)**:
   - ONLY include spoken human languages if explicitly requested (e.g., "Must speak French").
   - DO NOT default to "English". Return [] if not mentioned.

5. **domains & engineering_focus**:
   - **Domains**: Business or broad engineering areas (e.g., "Fintech", "Cybersecurity").
   - **Engineering Focus**: Architectural priorities (e.g., "Scalability", "Distributed Systems").
   - DO NOT infer these unless strongly implied by the query.

6. **competencies & behavioral_traits**:
   - **Competencies**: Measurable technical abilities (e.g., "System Design", "Debugging").
   - **Behavioral**: Soft skills (e.g., "Leadership", "Collaboration").

### RECRUITER QUERY:
\"\"\"
{user_query}
\"\"\"

IMPORTANT EXTRACTION RULES

ONLY extract concepts explicitly stated
or very directly implied by the recruiter.

DO NOT:
- expand semantic meaning
- infer related traits
- infer neighboring competencies
- generate ontology expansions
- add related behavioral qualities

BAD EXAMPLE:

Recruiter says:
"Need communication evaluation"

WRONG extraction:
["communication", "empathy", "customer focus"]

CORRECT extraction:
["communication"]

### GUARDRAIL: ROLE_TITLE VS. SEMANTIC_REQUIREMENT
A role_title MUST be a professional title (e.g., Engineer, Manager, Analyst). 
If the user provides a skill (e.g., 'Communication', 'Python', 'Leadership'), 
place it in the matching 'semantic_requirements' category and leave 'role_titles' EMPTY.

Example:
User: "I need a leadership assessment."
Result: role_titles: [], behavioral_traits: ["leadership"]

### EXTRACTION EXAMPLES:

- INPUT: "Need communication evaluation"
  OUTPUT: {{ "role_titles": [], "semantic_requirements": {{ "behavioral_traits": ["communication"] }} }}

- INPUT: "Java Developer"
  OUTPUT: {{ "role_titles": ["Java Developer"], "semantic_requirements": {{ "technical_skills": ["Java"] }} }}

- INPUT: "Expert in Leadership and Python"
  OUTPUT: {{ "role_titles": [], "semantic_requirements": {{ "technical_skills": ["Python"], "behavioral_traits": ["Leadership"] }} }}

  

Extraction must remain minimal,
precise, and recruiter-grounded.

### OUTPUT STRUCTURE:
Return ONLY this JSON structure:
{{
  "role_titles": [],
  "explicit_constraints": {{
    "job_levels": [],
    "languages": [],
    "remote": null,
    "adaptive": null,
    "duration": null
  }},
  "semantic_requirements": {{
    "technical_skills": [],
    "domains": [],
    "competencies": [],
    "behavioral_traits": [],
    "engineering_focus": []
  }}
}} """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        response_format={"type": "json_object"}, 
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    # Clean Markdown if the model ignored the "No markdown" instruction
    if content.startswith("```"):
        content = re.sub(r'^```json\s*|```$', '', content, flags=re.MULTILINE).strip()

    return json.loads(content)


def extract_incremental_intent(
    user_query,
    current_state,
    dimension_context
):

    prompt = f"""
Return ONLY valid JSON. No preamble, no markdown blocks.
You are a State-Aware Recruiter Assistant. Your goal is to extract updates from a new message and apply them to the existing conversation state.

### CURRENT CONVERSATION STATE:
{json.dumps(current_state, indent=2)}

### AVAILABLE SEMANTIC DIMENSIONS:
{dimension_context}

### EXTRACTION & STATE RULES:
1. **Operation Logic**:
   - `append`: Add new requirements to existing lists.
   - `replace`: If the recruiter explicitly changes a requirement (e.g., "Actually, let's look for Python instead of Java").
   - `remove`: If the recruiter explicitly strikes a requirement.
2. **Atomic Extraction**: Extract ONLY what is explicitly stated. Do not infer "empathy" from "communication".
3. **Strict Constraints**: 
   - **job_levels**: ONLY extract if seniority keywords (Senior, Junior, etc.) are present.
   - **languages**: Human languages only. DO NOT default to English.
4. **No Re-generation**: Do not return the entire state. Return only the delta (the changes) found in the new message.

### CATEGORY DEFINITIONS:
- **role_titles**: Specific job titles only.
- **technical_skills**: Languages, frameworks, tools.
- **domains**: Industry/Business areas (Fintech, Healthcare).
- **engineering_focus**: Architectural priorities (Scalability, High Performance).
- **competencies/behavioral**: Technical abilities vs. soft skills.

### RECRUITER MESSAGE:
\"\"\"
{user_query}
\"\"\"

IMPORTANT EXTRACTION RULES

ONLY extract concepts explicitly stated
or very directly implied by the recruiter.

DO NOT:
- expand semantic meaning
- infer related traits
- infer neighboring competencies
- generate ontology expansions
- add related behavioral qualities

role_titles
role_titles must represent actual hiring job roles.

NEVER place:
- soft skills
- communication traits
- competencies
- technologies
- engineering domains

Examples of INVALID role_titles:
- communication
- leadership
- teamwork
- Java
- cloud infrastructure


BAD EXAMPLE:

Recruiter says:
"Need communication evaluation"

WRONG extraction:
["communication", "empathy", "customer focus"]

CORRECT extraction:
["communication"]

Extraction must remain minimal,
precise, and recruiter-grounded.

### OUTPUT STRUCTURE:
Return ONLY this JSON format:
{{
  "operation": "append", # append | replace | remove
  "role_titles": [],
  "explicit_constraints": {{
    "job_levels": [],
    "remote": null, 
    "languages": []
  }},
  "semantic_requirements": {{
    "technical_skills": [],
    "domains": [],
    "competencies": [],
    "behavioral_traits": [],
    "engineering_focus": []
  }}
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    return json.loads(content)

def merge_state(current_state,
                new_intent):

    # ---------------------------------
    # Role titles
    # ---------------------------------

    for role in new_intent.get(
        "role_titles",
        []
    ):

        if role not in current_state[
            "role_titles"
        ]:

            current_state[
                "role_titles"
            ].append(role)

    # ---------------------------------
    # Explicit constraints
    # ---------------------------------

    explicit_current = current_state[
        "explicit_constraints"
    ]

    explicit_new = new_intent[
        "explicit_constraints"
    ]

    for key, value in explicit_new.items():

        if value is None:
            continue

        # Replace scalar values
        if isinstance(value, bool) \
           or isinstance(value, str):

            explicit_current[key] = value

        # Merge list values
        elif isinstance(value, list):

            for item in value:

                if item not in explicit_current[key]:

                    explicit_current[key].append(
                        item
                    )

    # ---------------------------------
    # Semantic requirements
    # ---------------------------------

    semantic_current = current_state[
        "semantic_requirements"
    ]

    semantic_new = new_intent[
        "semantic_requirements"
    ]

    for category, values in semantic_new.items():

        for value in values:

            normalized = normalize_term(value)

            if normalized not in semantic_current[
                category
            ]:

                semantic_current[
                    category
                ].append(normalized)

    return current_state


def apply_state_mutation(
    current_state,
    delta
):

    operation = delta.get(
        "operation",
        "append"
    )

    # ---------------------------------
    # Role titles
    # ---------------------------------

    for role in delta.get("role_titles", []):

        if operation == "append":

            if role not in current_state[
                "role_titles"
            ]:

                current_state[
                    "role_titles"
                ].append(role)

        elif operation == "replace":

            current_state[
                "role_titles"
            ] = [role]

    # ---------------------------------
    # Explicit constraints
    # ---------------------------------

    explicit_current = current_state["explicit_constraints"]

    # Use .get() to avoid KeyError if the LLM omits the key
    explicit_delta = delta.get("explicit_constraints", {})

    for key, value in explicit_delta.items():

        if value is None:
            continue

        if isinstance(value, bool):

            explicit_current[key] = value

        elif isinstance(value, str):

            explicit_current[key] = value

        elif isinstance(value, list):

            if operation == "replace":

                explicit_current[key] = value

            elif operation == "append":

                for item in value:

                    if item not in explicit_current[key]:

                        explicit_current[key].append(
                            item
                        )

            elif operation == "remove":

                explicit_current[key] = [
                    item
                    for item in explicit_current[key]
                    if item not in value
                ]

    # ---------------------------------
    # Semantic requirements
    # ---------------------------------

    semantic_current = current_state[
        "semantic_requirements"
    ]

    semantic_delta = delta.get(
    "semantic_requirements",
    {}
    )

    for category, values in semantic_delta.items():

        normalized_values = [
            normalize_term(v)
            for v in values
        ]

        if operation == "replace":

            semantic_current[
                category
            ] = normalized_values

        elif operation == "append":

            for value in normalized_values:

                if value not in semantic_current[
                    category
                ]:

                    semantic_current[
                        category
                    ].append(value)

        elif operation == "remove":

            semantic_current[
                category
            ] = [
                item
                for item in semantic_current[
                    category
                ]
                if item not in normalized_values
            ]

    return current_state

def build_search_query(intent):
    parts = []
    
    # 1. Add Role Titles (Highest Priority)
    # This ensures the core job title is the first thing the embedding model sees
    parts.extend(intent.get("role_titles", []))

    # 2. Add Explicit Constraints
    explicit = intent.get("explicit_constraints", {})
    for value in explicit.values():
        if isinstance(value, list):
            parts.extend([str(v) for v in value])
        elif isinstance(value, str) and value:
            parts.append(value)

    # 3. Add High-Signal Semantic Requirements
    semantic = intent.get("semantic_requirements", {})
    high_signal_fields = [
        "technical_skills",
        "domains",
        "competencies",
        "engineering_focus"
    ]

    for field in high_signal_fields:
        values = semantic.get(field, [])
        if isinstance(values, list):
            parts.extend([str(v) for v in values])

    # Clean up: Remove duplicates and join
    unique_parts = list(dict.fromkeys(parts)) 
    return " ".join(unique_parts)


# Load the model once at the top of your script
# This model produces exactly 384 dimensions
print("Loading local 384-dim embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text, is_query=True):
    try:
        # Arctic-embed models perform best when queries are prefixed 
        # with this specific instruction
        if is_query:
            text = f"Represent this sentence for searching relevant passages: {text}"
            
        embedding = model.encode(text, normalize_embeddings=True)
        return np.array(embedding).astype('float32').reshape(1, -1)
    except Exception as e:
        print(f"DEBUG ERROR: Embedding failed: {e}")
        return None

def semantic_search(query, index, metadata, top_k=5):
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        print("Search failed: Could not generate embedding.")
        return []

    # FAISS search
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(distances[0], indices[0]):
        # Prevent index out of bounds if FAISS returns -1
        if idx < 0 or idx >= len(metadata):
            continue
            
        doc = metadata[idx]
        results.append({
            "score": float(score),
            "name": doc["name"],
            "url": doc["url"],
            "metadata": doc["metadata"],
            "semantic_dimensions": doc.get("semantic_dimensions", {})
        })

    return results

def normalize_term(term):

    term = term.lower().strip()

    normalization_map = {

        # Communication

        "good communication skill": "communication",

        "communication skills": "communication",

        # Debugging

        "debugging ability": "debugging",

        # Problem solving

        "problem-solving": "problem solving",

        "problem solving ability": "problem solving",

        # Leadership

        "leadership skills": "leadership",

        # Collaboration

        "team collaboration": "collaboration"

    }

    return normalization_map.get(term, term)


def expand_roles(intent):

    role_titles = intent.get(
        "role_titles",
        []
    )

    semantic = intent[
        "semantic_requirements"
    ]

    for role in role_titles:

        normalized_role = normalize_term(role)

        for known_role, expansion in (
            ROLE_EXPANSION_MAP.items()
        ):

            if known_role in normalized_role:

                for category, values in (
                    expansion.items()
                ):

                    existing = set(
                        semantic.get(
                            category,
                            []
                        )
                    )

                    existing.update(values)

                    semantic[category] = (
                        list(existing)
                    )

    return intent


def normalize_semantic_requirements(
    intent
):

    semantic = intent[
        "semantic_requirements"
    ]

    for category, values in (
        semantic.items()
    ):

        normalized_values = []

        for value in values:

            normalized = (
                NORMALIZATION_MAP.get(
                    normalize_term(value),
                    value
                )
            )

            normalized_values.append(
                normalized
            )

        # remove duplicates
        semantic[category] = list(
            set(normalized_values)
        )

    return intent

def compute_semantic_overlap(intent_semantics,
                             assessment_semantics):

    overlap_score = 0

    overlap_details = []

    categories = [
        "technical_skills",
        "domains",
        "competencies",
        "behavioral_traits",
        "engineering_focus"
    ]

    for category in categories:

        recruiter_values = [
            normalize_term(v)
            for v in intent_semantics.get(
                category,
                []
            )
        ]

        assessment_values = [
            normalize_term(v)
            for v in assessment_semantics.get(
                category,
                []
            )
        ]

        overlap = set(recruiter_values).intersection(
            set(assessment_values)
        )

        if overlap:

            score = len(overlap)

            overlap_score += score

            overlap_details.append({
                "category": category,
                "matches": list(overlap),
                "score": score
            })

    return overlap_score, overlap_details

def validate_constraints(intent, results):

    validated_results = []

    explicit = intent["explicit_constraints"]

    for result in results:

        metadata = result["metadata"]

        compatibility_score = 0

        validation_notes = []

        # -----------------------------
        # Adaptive validation
        # -----------------------------

        if explicit["adaptive"] is True:

            if metadata.get("adaptive") == "yes":
                compatibility_score += 2
            else:
                compatibility_score -= 3
                validation_notes.append(
                    "adaptive mismatch"
                )

        # -----------------------------
        # Remote validation
        # -----------------------------

        if explicit["remote"] is True:

            if metadata.get("remote") == "yes":
                compatibility_score += 2
            else:
                compatibility_score -= 3
                validation_notes.append(
                    "remote mismatch"
                )

        # -----------------------------
        # Job level validation
        # -----------------------------

        requested_levels = explicit.get(
            "job_levels",
            []
        )

        metadata_levels = metadata.get(
            "job_levels",
            []
        )

        if requested_levels:

            overlap = set(requested_levels).intersection(
                set(metadata_levels)
            )

            if overlap:
                compatibility_score += 2
            else:
                compatibility_score -= 2
                validation_notes.append(
                    "job level mismatch"
                )
        
        # -----------------------------
        # Semantic compatibility
        # -----------------------------

        semantic_overlap_score, overlap_details = (
            compute_semantic_overlap(
                intent["semantic_requirements"],
                result["semantic_dimensions"]
            )
        )

        compatibility_score += semantic_overlap_score

        result["semantic_overlap"] = (
            overlap_details
        )

        # -----------------------------
        # Final score
        # -----------------------------

        result["compatibility_score"] = (
            compatibility_score
        )

        result["validation_notes"] = (
            validation_notes
        )

        # Lower distance is better
        semantic_score = -result["score"]

        exact_match = compute_exact_match_score(
            intent,
            result
        )

        compatibility_score += exact_match[
            "score"
        ]

        result["final_score"] = (

            compute_weighted_rerank_score(

                result=result,

                exact_match=exact_match,

                compatibility_score=compatibility_score,

                distance=semantic_score

            )

        )

        # Store exact match details
        result["exact_match"] = exact_match

        validated_results.append(result)

    validated_results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return validated_results


def compute_exact_match_score(
    intent,
    result
):

    score = 0

    semantic = intent[
        "semantic_requirements"
    ]

    result_semantics = result.get(
        "semantic_dimensions",
        {}
    )

    # ---------------------------------
    # Technical skills (HIGH weight)
    # ---------------------------------

    requested_skills = set(
        normalize_term(skill)
        for skill in semantic.get(
            "technical_skills",
            []
        )
    )

    result_skills = set(
        normalize_term(skill)
        for skill in result_semantics.get(
            "technical_skills",
            []
        )
    )

    exact_skill_matches = (
        requested_skills &
        result_skills
    )

    score += len(
        exact_skill_matches
    ) * 5

    # ---------------------------------
    # Domains (MEDIUM weight)
    # ---------------------------------

    requested_domains = set(
        normalize_term(domain)
        for domain in semantic.get(
            "domains",
            []
        )
    )

    result_domains = set(
        normalize_term(domain)
        for domain in result_semantics.get(
            "domains",
            []
        )
    )

    exact_domain_matches = (
        requested_domains &
        result_domains
    )

    score += len(
        exact_domain_matches
    ) * 3

    # ---------------------------------
    # Competencies (LOWER weight)
    # ---------------------------------

    requested_competencies = set(
        normalize_term(c)
        for c in semantic.get(
            "competencies",
            []
        )
    )

    result_competencies = set(
        normalize_term(c)
        for c in result_semantics.get(
            "competencies",
            []
        )
    )

    competency_matches = (
        requested_competencies &
        result_competencies
    )

    # ---------------------------------
    # Behavioral traits
    # ---------------------------------

    requested_traits = set(
        normalize_term(trait)
        for trait in semantic.get(
            "behavioral_traits",
            []
        )
    )

    result_traits = set(
        normalize_term(trait)
        for trait in result_semantics.get(
            "behavioral_traits",
            []
        )
    )

    trait_matches = (
        requested_traits &
        result_traits
    )

    score += len(
        trait_matches
    ) * 2

    return {
        "score": score,

        "exact_skill_matches":
            list(exact_skill_matches),

        "exact_domain_matches":
            list(exact_domain_matches),

        "competency_matches":
            list(competency_matches)
    }

def compute_weighted_rerank_score(
    result,
    exact_match,
    compatibility_score,
    distance
):

    score = 0

    # ---------------------------------
    # Exact semantic matches
    # ---------------------------------

    score += exact_match["score"]

    # ---------------------------------
    # Constraint compatibility
    # ---------------------------------

    score += compatibility_score * 2

    # ---------------------------------
    # Behavioral / personality bonus
    # ---------------------------------

    keys = result["metadata"].get(
        "keys",
        []
    )

    if (
        "Personality & Behavior"
        in keys
    ):

        score += 4

    # ---------------------------------
    # Competency assessments
    # ---------------------------------

    if (
        "Competencies"
        in keys
    ):

        score += 3

    # ---------------------------------
    # Adaptive bonus
    # ---------------------------------

    if (
        result["metadata"].get(
            "adaptive"
        ) == "yes"
    ):

        score += 1

    # ---------------------------------
    # Penalize semantic distance
    # ---------------------------------

    score -= distance * 2

    return round(score, 4)

def estimate_confidence(intent,
                        validated_results):

    confidence_score = 0

    reasoning = []

    top_result = validated_results[0]

    # ---------------------------------
    # Strong semantic match
    # ---------------------------------

    if top_result["final_score"] > 3:

        confidence_score += 2

        reasoning.append(
            "strong top recommendation score"
        )

    # ---------------------------------
    # Constraint mismatches
    # ---------------------------------

    mismatch_count = len(
        top_result["validation_notes"]
    )

    if mismatch_count == 0:

        confidence_score += 2

        reasoning.append(
            "no constraint mismatches"
        )

    else:

        confidence_score -= mismatch_count

    # ---------------------------------
    # Semantic coverage
    # ---------------------------------

    recruiter_semantics = (
        intent["semantic_requirements"]
    )

    overlap_categories = {
        item["category"]
        for item in top_result[
            "semantic_overlap"
        ]
    }

    missing_categories = []

    for category, values in recruiter_semantics.items():

        if values and category not in overlap_categories:

            missing_categories.append(category)

    if not missing_categories:

        confidence_score += 2

        reasoning.append(
            "all semantic categories covered"
        )

    else:

        confidence_score -= len(
            missing_categories
        )

    return {
        "confidence_score": confidence_score,
        "missing_categories": missing_categories,
        "reasoning": reasoning
    }



def analyze_coverage(
    intent,
    validated_results
):

    top_result = validated_results[0]

    recruiter_semantics = (
        intent["semantic_requirements"]
    )

    overlap_categories = {
        item["category"]
        for item in top_result[
            "semantic_overlap"
        ]
    }

    coverage_gaps = []

    for category, values in recruiter_semantics.items():

        # recruiter explicitly provided category
        if values:

            # but retrieval lacks overlap
            if category not in overlap_categories:

                coverage_gaps.append(category)

    return coverage_gaps


def generate_reasoning_response(
    coverage_gaps
):

    responses = []

    reasoning_map = {

        "behavioral_traits":
            (
                "I found strong technical assessment coverage, "
                "but limited behavioral or communication-focused coverage. "
                "You may want to combine technical and behavioral assessments."
            ),

        "domains":
            (
                "The requested engineering domain has limited direct assessment coverage. "
                "You may want to use broader engineering evaluations alongside domain-specific screening."
            ),

        "competencies":
            (
                "Some requested competencies are not strongly represented in the current assessment matches."
            ),

        "engineering_focus":
            (
                "Specialized engineering focus areas may require combining multiple assessments."
            )
    }

    for gap in coverage_gaps:

        if gap in reasoning_map:

            responses.append(
                reasoning_map[gap]
            )

    return responses

def generate_clarification(intent, validated_results, missing_categories):
    if not missing_categories:
        return []

    # 1. Gather candidate suggestions from the top 5 search results
    context_suggestions = {}
    for category in missing_categories:
        seen = set()
        for res in validated_results[:5]:
            # Pull values from the actual assessment data found by the vector search
            dims = res.get("semantic_dimensions", {}).get(category, [])
            for d in dims:
                seen.add(d)
        if seen:
            context_suggestions[category] = list(seen)

    # 2. Prompt the LLM to bridge the gap
    prompt = f"""
Return ONLY valid JSON for the data extraction, and a natural string for the question.
You are a Proactive Recruiter Assistant.

### CONTEXT:
1. **Recruiter Intent**: {json.dumps(intent['semantic_requirements'])}
2. **Missing Categories**: {missing_categories}
3. **Available Database Matches**: {json.dumps(context_suggestions, indent=2)}

### TASK:
Generate a single, high-signal clarification question that maps the recruiter's vague intent to our specific assessment attributes.

### GUIDELINES:
- **Map, Don't Just Ask**: If the user wants "Backend" and we have "Distributed Systems", ask: "Would you like the technical assessment to focus specifically on distributed systems and high-performance architecture?"
- **Constraint Priority**: Use the following logic for missing data:
    - **Behavioral**: Ask if they need soft-skill evaluation (e.g., "communication-focused evaluation").
    - **Competencies**: Ask if they want to prioritize "debugging" or "system design".
    - **Domains**: Ask for specific industry focus (e.g., "Fintech" vs "Cloud Infrastructure").
- **Constraint**: No more than 2-3 specific suggestions per question. Keep it under 25 words.

CRITICAL RESPONSE RULES

Return ONLY:
- one short recruiter-facing clarification question
- plain text only

DO NOT:
- explain reasoning
- output JSON
- use markdown
- summarize intent
- mention database matches
- mention extraction
- mention categories

Your response must look like a natural recruiter conversation.


### CLARIFICATION TEMPLATES (Internal Logic):
{{
    "behavioral_traits": "Do you also want behavioral or communication-focused evaluation alongside the technical assessment?",
    "competencies": "Should the assessment emphasize specific competencies like problem solving or analytical reasoning?",
    "engineering_focus": "Are you looking for specialized focus areas such as cloud infrastructure or distributed systems?",
    "domains": "Can you clarify the primary engineering or business domain for this role?"
}}

### QUESTION:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    return [response.choices[0].message.content.strip()]

def main():

    print("\nLoading retrieval system...")

    global conversation_state

    index, metadata = load_index()

    dimensions = load_dimensions()

    dimension_context = build_dimension_context(
        dimensions
    )

    print("\nSemantic retrieval system ready.")

    while True:

        user_query = input(
            "\nRecruiter Query: "
        )

        if user_query.lower() == "exit":
            break

        try:

            print("\nExtracting intent...")

            # ---------------------------------
            # First turn → full extraction
            # ---------------------------------

            if is_empty_state(
                conversation_state
            ):

                print(
                    "\nUsing full intent extraction..."
                )

                extracted = extract_intent(
                    user_query,
                    dimension_context
                )

                conversation_state = merge_state(
                    conversation_state,
                    extracted
                )

            # ---------------------------------
            # Later turns → incremental extraction
            # ---------------------------------

            else:

                print(
                    "\nUsing incremental extraction..."
                )

                delta = extract_incremental_intent(
                    user_query,
                    conversation_state,
                    dimension_context
                )

                conversation_state = apply_state_mutation(
                    conversation_state,
                    delta
                )

            intent = conversation_state

            intent = expand_roles(intent)

            intent = normalize_semantic_requirements(
                intent
            )

            print("\n=== CONVERSATION STATE ===\n")

            print(json.dumps(intent, indent=2))

            search_query = build_search_query(
                intent
            )

            print("\n=== SEARCH QUERY ===\n")

            print(search_query)

            results = semantic_search(
                search_query,
                index,
                metadata
            )

            validated_results = validate_constraints(
                intent,
                results
            )

            confidence = estimate_confidence(
                intent,
                validated_results
            )

            coverage_gaps = analyze_coverage(
                intent,
                validated_results
            )

            print("\n=== COVERAGE GAPS ===\n")

            print(coverage_gaps)

            reasoning_responses = (
                generate_reasoning_response(
                    coverage_gaps
                )
            )

            if reasoning_responses:

                print("\n=== RECOMMENDATION REASONING ===\n")

                for response in reasoning_responses:

                    print(f"- {response}")

            print("\n=== CONFIDENCE ===\n")

            print(json.dumps(
                confidence,
                indent=2
            ))

            if confidence["confidence_score"] < 4:

                clarification_questions = generate_clarification(
                    conversation_state, 
                    validated_results, 
                    confidence["missing_categories"]
                )

                clarification = clarification_questions

                print("\n=== CLARIFICATION QUESTIONS ===\n")

                for question in clarification:

                    print(f"- {question}")

            if not validated_results:
                print("DEBUG: Search completed but returned 0 results.")
            else:
                print(f"DEBUG: Found {len(validated_results)} matches.")

            print("\n=== TOP MATCHES ===\n")

            for idx, (original, validated) in enumerate(
                zip(results, validated_results),
                start=1
            ):

                print(f"\n#{idx}")

                print(
                    f"Name: {original['name']}"
                )

                print(
                    f"Distance Score: {original['score']:.4f}"
                )

                print(
                    f"URL: {original['url']}"
                )

                print(
                    f"Metadata: {original['metadata']}"
                )
                print(
                    f"Compatibility Score: "
                    f"{validated['compatibility_score']}"
                )

                print(
                    f"Final Score: "
                    f"{validated['final_score']:.4f}"
                )

                print(
                    f"Validation Notes: "
                    f"{validated['validation_notes']}"
                )

                print(
                    f"Semantic Overlap: "
                    f"{validated['semantic_overlap']}"
                )

                print(
                    "Exact Match:",
                    validated["exact_match"]
                )

        except json.JSONDecodeError as e:
            print(f"\nCRITICAL ERROR (JSON): The LLM returned invalid JSON. {e}")
        except KeyError as e:
            print(f"\nCRITICAL ERROR (KeyError): Expected key missing in data: {e}")
        except Exception as e:
            import traceback
            print("\n--- UNEXPECTED CRASH ---")
            traceback.print_exc() # This prints the EXACT line number and file
            print("------------------------")


if __name__ == "__main__":
    main()
