import os
import json
from groq import Groq

# Initialize the client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_groq_json(system_prompt, user_prompt, model="llama-3.3-70b-versatile"):
    """
    Helper to force Groq to return a valid JSON object.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON."},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling Groq JSON: {e}")
        return {}

def call_groq_completion(system_prompt, user_prompt, model="llama-3.3-70b-versatile"):
    """
    Standard text completion for non-structured tasks.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq: {e}")
        return ""
