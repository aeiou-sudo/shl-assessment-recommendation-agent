import json
import time
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from groq import Groq

import os


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

CATALOG_PATH = Path("data/raw/shl_product_catalog.json")
OUTPUT_PATH = Path("data/processed/semantic_dimensions.json")


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f, strict=False)


def extract_semantics(description, model):
    prompt = f"""
    You are an expert at mapping technical assessments to recruiter-relevant dimensions.
    Extract dimensions based ONLY on the provided description. 
    Avoid vague categories like 'technology' or 'business'.

    Return ONLY valid JSON in this structure:
    {{
      "technical_skills": ["specific tools/languages"],
      "domains": ["engineering/business areas"],
      "competencies": ["measurable capabilities"],
      "behavioral_traits": ["interpersonal traits if explicit"],
      "engineering_focus": ["specialized priorities e.g. low-latency"]
    }}

    Assessment Description:
    \"\"\"
    {description}
    \"\"\"
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    
    # NEW: Handle empty responses or markdown formatting
    if not content or not content.strip():
        print("Warning: Received empty content from API")
        return {"technical_skills": [], "domains": [], "competencies": [], "behavioral_traits": [], "engineering_focus": []}

    # Clean markdown code blocks if the model included them
    clean_json = content.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(clean_json)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing failed for: {content[:100]}...")
        raise e

def main():

    catalog = load_catalog()

    aggregated_dimensions = defaultdict(set)

    processed_results = []

    model = "llama-3.1-8b-instant"

    for idx, item in enumerate(catalog):

        description = item.get("description", "")

        if not description.strip():
            continue

        print(f"\nProcessing {idx + 1}/{len(catalog)}")
        print(item.get("name"))

        try:

            semantics = extract_semantics(description, model)

            processed_results.append({
                "name": item.get("name"),
                "semantics": semantics
            })

            for category, values in semantics.items():

                if isinstance(values, list):

                    for value in values:
                        aggregated_dimensions[category].add(
                            value.strip().lower()
                        )

            time.sleep(1)

        except Exception as e:
            if "429" in str(e):
                print("Rate limit hit. Sleeping for 3 minutes...")
                time.sleep(180)
                model = "mixtral-8x7b-32768"

            else:
                 print(f"ERROR: {e}")

    output = {
        "aggregated_dimensions": {
            k: sorted(list(v))
            for k, v in aggregated_dimensions.items()
        },
        "assessment_semantics": processed_results
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved semantic dimensions to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
