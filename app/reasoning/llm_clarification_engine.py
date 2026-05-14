import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # This line loads the variables from .env

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are an expert conversational reasoning engine for hiring assessment recommendation systems.

Your task is to identify the SINGLE most critical "Identity Crisis" in the retrieved candidates that prevents a confident recommendation.

PRIORITIZATION HIERARCHY:
1. ROLE AMBIGUITY: Are the results split between different jobs (e.g., Data Science vs. Backend)?
2. SENIORITY AMBIGUITY: Are results split between different levels (e.g., Junior vs. Manager)?
3. DOMAIN AMBIGUITY: Is the industry context unclear?
4. TECHNOLOGY AMBIGUITY: Only focus here if roles and seniority are clear but tools vary wildly.

Guidelines:
- Focus on the dominant ambiguity that would most confuse a hiring manager.
- The clarification question MUST directly help distinguish between the strongest competing candidate interpretations.
- Avoid broad exploratory questions.
- Prefer questions that narrow the candidate space decisively.
- The question should ideally contrast the competing role interpretations explicitly when possible.

IMPORTANT RULES:
- Generate ONE concise, professional clarification question.
- DO NOT mention statistical distributions, data, or counts.
- DO NOT list options like a robot.
- Return STRICT JSON ONLY.

Required JSON format:
{
  "dominant_ambiguity": "...",
  "reasoning": "...",
  "clarification_question": "..."
}
"""

def filter_top_signals(distribution, limit=5):
    """Helper to remove noise and keep only high-frequency signals."""
    return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:limit])

def generate_clarification(query, analysis):
    # -----------------------------------------
    # Build compact, high-signal context
    # -----------------------------------------
    ambiguity_context = {
        "technology_distribution": filter_top_signals(analysis.get("technology_distribution", {})),
        "domain_distribution": filter_top_signals(analysis.get("domain_distribution", {})),
        "role_signal_distribution": filter_top_signals(analysis.get("role_signal_distribution", {})),
        "seniority_distribution": filter_top_signals(analysis.get("seniority_distribution", {}))
    }

    user_prompt = f"""
USER QUERY:
{query}

SEMANTIC AMBIGUITY DISTRIBUTIONS:
{json.dumps(ambiguity_context, indent=2)}
"""

    # -----------------------------------------
    # Groq Implementation with Fallback
    # -----------------------------------------
    # Primary: 70B for high reasoning. Fallback: 8B for TPD safety.
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
    except Exception as e:
        if "rate_limit_exceeded" in str(e).lower():
            # Silent fallback to 8B if TPD limit is hit
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
        else:
            raise e

    content = response.choices[0].message.content
    return json.loads(content)
