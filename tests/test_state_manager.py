from pathlib import Path
import sys

# Import your previously created engine
# Ensure the path corresponds to your local project structure
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from app.state.state_manager import (
    StateManager
)

manager = StateManager()

operations = [

    {
        "op": "replace",
        "field": "role_focus",
        "value": "Backend Developer"
    },

    {
        "op": "add",
        "field": "technology_stack",
        "value": "Python"
    },

    {
        "op": "add",
        "field": "technology_stack",
        "value": "Django"
    },

    {
        "op": "replace",
        "field": "seniority",
        "value": "Mid-Professional"
    }
]

manager.apply_operations(
    operations
)

print(
    manager.get_state()
)
