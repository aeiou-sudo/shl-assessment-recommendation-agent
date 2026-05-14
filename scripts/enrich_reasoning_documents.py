import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from groq import Groq # Using Groq client for the rotation
from dotenv import load_dotenv

# Configuration
INPUT_FILE = Path("generated/reasoning_documents.json")
OUTPUT_FILE = Path("generated/reasoning_documents_enriched.json")

# dual model rotation to maximize RPM (Requests Per Minute)
MODELS = ["llama-3.1-8b-instant", "llama-3.1-8b-instant"]

load_dotenv()  # This line loads the variables from .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a high-precision Semantic Reasoning Engine for professional assessments. Your goal is to transform raw assessment metadata into a multi-dimensional technical profile for an automated recommendation system.

### CORE EXTRACTION DIMENSIONS:
1. TECHNOLOGIES: Specific tools, languages, frameworks, or software (e.g., Python, AWS, SAP).
2. CORE_COMPETENCIES: Fundamental skills or cognitive abilities being measured (e.g., Data Analysis, Logical Reasoning, Conflict Resolution).
3. DOMAINS & INDUSTRIES: The field of application or specialized industry context (e.g., Cybersecurity, Financial Services, Healthcare).
4. METHODOLOGIES: Working styles, frameworks, or operational standards (e.g., Agile, SDLC, Six Sigma, GAAP).
5. ROLE_SIGNALS: Specific job functions or titles this assessment aligns with (e.g., Backend Developer, Talent Acquisition, DevOps Engineer).
6. SENIORITY_SIGNALS: The level of expertise or organizational tier implied (e.g., Strategic Leadership, Graduate/Entry-Level, Professional Individual Contributor).
7. ASSESSMENT_INTENT: The primary objective—Is it for 'Talent Acquisition', 'Leadership Development', 'Technical Certification', or '360 Feedback'?

### EXTRACTION RULES:
- RULE 1: STRICT JSON ONLY. No conversational filler or introductory text.
- RULE 2: NO HALLUCINATION. If a dimension is not explicitly supported by the input, return an empty list [].
- RULE 3: SEMANTIC DENSITY. Prefer multi-word concepts (e.g., 'Relational Database Management' instead of 'Database').
- RULE 4: NOISE REDUCTION. Ignore generic filler such as "measures ability," "designed for," or "successfully complete."
- RULE 5: INFERENCE LIMITS. You may infer 'Role Signals' from technical skills (e.g., React implies Frontend Development), but do not invent specific job titles not suggested by the text.
- RULE 6: Do not infer organizational use-cases unless strongly supported by the input.

### GOOD EXAMPLES:

INPUT:
Name: .NET Framework 4.5
Description: Measures knowledge of developing applications using .NET 4.5, C#, and ASP.NET. Focuses on security, data access, and web services.
Categories: [Software, Information Technology]
Job Levels: [Mid-Professional, Professional Individual Contributor]

OUTPUT:
{
  "technologies": [".NET Framework 4.5", "C#", "ASP.NET"],
  "domains": ["Software Development", "Information Technology"],
  "competencies": ["Web Services Development", "Application Security", "Data Access Logic"],
  "methodologies": ["SDLC"],
  "role_signals": ["Backend Developer", "Full Stack Developer"],
  "seniority_signals": ["Mid-Professional"],
  "assessment_intent": ["Talent Acquisition"]
}

INPUT:
Name: SHL 360 Multi-Rater Feedback
Description: Provides a holistic view of an employee by gathering feedback from managers, peers, and direct reports based on the Universal Competency Framework.
Categories: [Development & 360, Personality & Behavior]
Job Levels: [Director, Manager, Executive]

OUTPUT:
{
  "technologies": [],
  "domains": ["Corporate Leadership", "Human Resources"],
  "competencies": ["Self-Awareness", "Professional Development", "Interpersonal Impact"],
  "methodologies": ["360-Degree Feedback", "Universal Competency Framework (UCF)"],
  "role_signals": ["People Manager", "Organizational Leader"],
  "seniority_signals": ["Director", "Manager", "Executive"],
  "assessment_intent": ["Leadership Development", "Performance Management"]
}
"""

def enrich_document(doc, model_index):
    user_prompt = f"""
    ASSESSMENT DATA:
    Name: {doc['name']}
    Description: {doc['description']}
    Categories: {", ".join(doc.get("categories", []))}
    Job Levels: {", ".join(doc.get("job_levels", []))}
    """

    model = MODELS[model_index % len(MODELS)]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content), model
    except Exception as e:
        raise e

def main():
    print(f"Loading documents from {INPUT_FILE}...")
    
    # Use strict=False to handle the control characters in your JSON
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.loads(f.read(), strict=False)

    # Check for existing progress to allow resuming
    enriched_docs = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            enriched_docs = json.load(f)
        print(f"Resuming from entry {len(enriched_docs)}...")

    # Process remaining docs
    for i in tqdm(range(len(enriched_docs), len(docs))):
        doc = docs[i]
        
        try:
            enriched_data, used_model = enrich_document(doc, i)
            
            final_doc = {
                **doc,
                "semantic_reasoning": enriched_data,
                "enrichment_metadata": {"model": used_model, "timestamp": time.time()}
            }
            
            enriched_docs.append(final_doc)

            # Checkpoint: Save every 10 items to prevent data loss
            if i % 10 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(enriched_docs, f, indent=2)
            
            # Tiny sleep to stay safe with Groq Free Tier RPM
            if i % 2 == 0:
                time.sleep(0.5)

        except Exception as e:
            print(f"\nCRITICAL FAILURE at '{doc['name']}': {e}")
            # Save progress before exiting
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(enriched_docs, f, indent=2)
            break

    # Final Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_docs, f, indent=2)
    
    print(f"\nDONE. Processed {len(enriched_docs)} documents.")

if __name__ == "__main__":
    main()
