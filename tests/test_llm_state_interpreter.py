from pathlib import Path
import sys

# Import your previously created engine
# Ensure the path corresponds to your local project structure
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from app.state.llm_state_interpreter import (
    interpret_user_message
)

from app.state.state_manager import (
    StateManager
)


manager = StateManager()

# -----------------------------------------
# Initial state
# -----------------------------------------

initial_operations = [

    {
        "op": "replace",
        "field": "role_focus",
        "value": "Backend Developer"
    },

    {
        "op": "add",
        "field": "technology_stack",
        "value": "Java"
    }
]

manager.apply_operations(
    initial_operations
)

# -----------------------------------------
# User update
# -----------------------------------------

user_message = (
    "Actually switch from Java to Python "
    "and include DevOps responsibilities."
)

result = interpret_user_message(
    manager.get_state(),
    user_message
)

print("\nLLM OUTPUT:\n")

print(result)

# -----------------------------------------
# Apply operations
# -----------------------------------------

manager.apply_operations(
    result["operations"]
)

print("\nUPDATED STATE:\n")

print(manager.get_state())
