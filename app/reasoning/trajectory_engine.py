from app.llm.groq_client import (
    call_groq_json
)


SYSTEM_PROMPT = """
You are an expert semantic trajectory analyzer for a conversational retrieval system.

Your task is to determine whether the NEW USER MESSAGE:

1. refines the existing semantic trajectory
2. pivots to a related trajectory
3. starts a completely new trajectory

--------------------------------------------------
TRAJECTORY DEFINITIONS
--------------------------------------------------

1. refinement
- preserves the same underlying intent
- adds constraints
- narrows scope
- clarifies requirements
- adds technologies, domains, competencies, or preferences

2. pivot
- partially preserves semantic continuity
- changes direction while remaining related
- modifies role/domain/technical trajectory
- introduces adjacent or competing objectives

3. hard_reset
- introduces a semantically unrelated objective
- invalidates most previous contextual assumptions
- starts a fundamentally different retrieval trajectory

--------------------------------------------------
IMPORTANT REASONING RULES
--------------------------------------------------

- Use semantic reasoning, not keyword matching.
- Evaluate continuity using:
    - intent continuity
    - objective continuity
    - semantic compatibility
    - contextual preservation
    - constraint preservation
    - retrieval trajectory continuity

- Do NOT assume any fixed ontology
  such as:
    - frontend/backend
    - engineering/management
    - technology categories
    - recruitment hierarchies

- Do NOT infer relationships unless
  semantically justified.

- Prefer semantic continuity over
  superficial token overlap.

- A trajectory is a retrieval objective,
  not a job title.

--------------------------------------------------
STATE-AWARE REASONING
--------------------------------------------------

You will receive:
- current conversation state
- new user message

The state may contain:
- structured constraints
- semantic memory
- retrieved trajectory signals
- historical clarifications

You must determine whether the new message:
- strengthens existing retrieval intent
- redirects retrieval intent
- or replaces retrieval intent entirely

--------------------------------------------------
DECISION GUIDELINES
--------------------------------------------------

Use "refinement" when:
- previous constraints remain valid
- new message specializes existing direction

Use "pivot" when:
- some prior constraints remain useful
- but retrieval direction changes meaningfully

Use "hard_reset" when:
- previous constraints become mostly irrelevant
- semantic retrieval objective changes entirely

If uncertain between:
- pivot vs hard_reset
→ prefer pivot

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY valid JSON.

{
    "trajectory_type": "pivot",
    "confidence": 0.91,
    "reasoning": "The new message partially preserves the existing semantic trajectory while redirecting the retrieval objective toward a related but distinct domain."
}
"""


def detect_trajectory_shift(
    current_state,
    new_message
):

    payload = {

        "current_state":
            current_state,

        "new_message":
            new_message
    }

    response = call_groq_json(

        system_prompt=SYSTEM_PROMPT,

        user_prompt=str(payload)
    )

    return response