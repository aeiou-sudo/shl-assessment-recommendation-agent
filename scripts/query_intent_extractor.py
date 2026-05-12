import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

import json
import re

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

DIMENSIONS_PATH = Path(
    "data/processed/semantic_dimensions.json"
)


def load_dimensions():

    with open(DIMENSIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f, strict=False)


def build_dimension_context(dimensions):

    aggregated = dimensions["aggregated_dimensions"]

    context = []

    for category, values in aggregated.items():

        sample_values = values[:30]

        context.append(
            f"{category}: {', '.join(sample_values)}"
        )

    return "\n".join(context)


def extract_intent(user_query, dimension_context):
    prompt = f"""
You are building a recruiter intent extraction system.
Your task is to convert recruiter hiring requests into structured semantic intent.

IMPORTANT RULES:
1. Return ONLY valid JSON.
2. Use ONLY dimensions relevant to the recruiter query.
3. Do NOT hallucinate information.
4. Normalize terms into concise recruiter-facing phrases.

AVAILABLE SEMANTIC DIMENSIONS:
{dimension_context}

Return ONLY this JSON structure:
{{
  "explicit_constraints": {{
    "job_levels": [],
    "remote": null,
    "adaptive": null,
    "duration": null,
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

Recruiter Query:
\"\"\"
{user_query}
\"\"\"
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content

    try:
        # 1. Isolate the JSON object using regex (finds everything between the first and last curly braces)
        # This ignores markdown backticks or "Here is the JSON:" text.
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            clean_content = json_match.group(0)
            
            # 2. Standardize whitespace (replaces non-breaking spaces \xa0 with standard spaces)
            clean_content = clean_content.replace('\xa0', ' ')
            
            return json.loads(clean_content)
        else:
            raise ValueError("No JSON object found in response")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"--- Parsing Error for Query: {user_query[:50]}... ---")
        print(f"Raw Output: {content[:100]}...")
        
        # Return a 'safe' empty structure so your script keeps running
        return {
            "explicit_constraints": {"job_levels": [], "remote": None, "adaptive": None, "duration": None, "languages": []},
            "semantic_requirements": {"technical_skills": [], "domains": [], "competencies": [], "behavioral_traits": [], "engineering_focus": []}
        }


def main():

    dimensions = load_dimensions()

    dimension_context = build_dimension_context(dimensions)

    print("\n=== Recruiter Query Intent Extraction ===\n")

    while True:

        user_query = input("\nRecruiter Query: ")

        if user_query.lower() == "exit":
            break

        try:

            intent = extract_intent(
                user_query,
                dimension_context
            )

            print("\n=== STRUCTURED INTENT ===\n")

            print(json.dumps(intent, indent=2))

        except Exception as e:

            print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
