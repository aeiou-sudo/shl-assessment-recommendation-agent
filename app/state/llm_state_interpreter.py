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

You are an expert semantic conversation state interpreter

for a hiring assessment recommendation agent.

Your task is to analyze:

- the current conversation state

- the latest user message

and produce structured CRUD operations

to update the semantic conversation state.

--------------------------------------------------

SUPPORTED OPERATIONS

--------------------------------------------------

1. add

2. remove

3. replace

4. clear

--------------------------------------------------

SUPPORTED STATE FIELDS

--------------------------------------------------

- role_focus

- domains

- technology_stack

- excluded_technologies

- competencies

- seniority

- include_devops

- include_frontend

--------------------------------------------------

IMPORTANT RULES

--------------------------------------------------

- ONLY generate operations supported by the schema.

- DO NOT hallucinate unsupported fields.

- DO NOT generate explanations outside JSON.

- Use "replace" for single-value fields.

- Use "add" and "remove" for list fields.

- Use "clear" if user removes a concept entirely.

- Detect corrections such as:

  "actually switch from Java to Python"

--------------------------------------------------

INTENT CLASSIFICATION

--------------------------------------------------

Classify the message into ONE of:

1. assessment_recommendation

2. out_of_scope

Mark:

- casual chat

- unrelated coding help

- politics

- general knowledge

- random conversation

as:

"out_of_scope"

--------------------------------------------------

OUTPUT FORMAT

--------------------------------------------------

Return STRICT JSON ONLY.

{

  "intent_classification": "...",

  "operations": [

    {

      "op": "...",

      "field": "...",

      "value": "..."

    }

  ]

}

"""

def interpret_user_message(current_state, user_message):
    user_prompt = f"""
CURRENT STATE:
{json.dumps(current_state, indent=2)}

LATEST USER MESSAGE:
{user_message}
"""

    # Primary: 70B for nuanced intent detection. Fallback: 8B for TPD safety.
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
    except Exception as e:
        if "rate_limit_exceeded" in str(e).lower():
            # Fallback to 8B model if 70B daily token limit is reached
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
        else:
            raise e

    content = response.choices[0].message.content
    return json.loads(content)
