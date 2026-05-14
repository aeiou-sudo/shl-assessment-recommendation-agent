from app.state.conversation_state import (
    ConversationState
)


class StateManager:

    def __init__(self):

        self.state = ConversationState()

    # -----------------------------------------
    # Generic state field groups
    # -----------------------------------------

    TRANSIENT_FIELDS = {

        "primary_intent",
        "intent_history",
        "candidate_history",
        "clarification_history",
        "clarification_turns"
    }

    CONTEXT_FIELDS = {

        "role_focus",
        "domains",
        "technology_stack",
        "excluded_technologies",
        "competencies",
        "seniority"
    }

    PERSISTENT_FIELDS = {

        "user_messages"
    }

    # -----------------------------------------
    # Generic reset utility
    # -----------------------------------------

    def reset_fields(
        self,
        fields
    ):

        for field in fields:

            if not hasattr(
                self.state,
                field
            ):
                continue

            current = getattr(
                self.state,
                field
            )

            if isinstance(
                current,
                list
            ):

                setattr(
                    self.state,
                    field,
                    []
                )

            else:

                setattr(
                    self.state,
                    field,
                    None
                )

    # -----------------------------------------
    # Soft Reset
    # Preserve broader trajectory memory
    # Remove contextual ambiguity
    # -----------------------------------------

    def soft_reset(self):

        reset_targets = (

            self.CONTEXT_FIELDS
            |
            self.TRANSIENT_FIELDS
        )

        self.reset_fields(
            reset_targets
        )

    # -----------------------------------------
    # Hard Reset
    # Full trajectory reset
    # -----------------------------------------

    def hard_reset(self):

        reset_targets = set()

        for field in vars(self.state):

            if field.startswith("_"):

                continue

            if field in self.PERSISTENT_FIELDS:

                continue

            reset_targets.add(field)

        self.reset_fields(
            reset_targets
        )

    # -----------------------------------------
    # Apply operations
    # -----------------------------------------

    def apply_operations(
        self,
        operations
    ):

        for operation in operations:

            op = operation.get("op")

            field = operation.get("field")

            value = operation.get("value")

            if not hasattr(
                self.state,
                field
            ):
                continue

            # -----------------------------------------
            # Replace
            # -----------------------------------------

            if op == "replace":

                setattr(
                    self.state,
                    field,
                    value
                )

            # -----------------------------------------
            # Add
            # -----------------------------------------

            elif op == "add":

                current = getattr(
                    self.state,
                    field
                )

                if isinstance(
                    current,
                    list
                ):

                    if value not in current:

                        current.append(value)

            # -----------------------------------------
            # Remove
            # -----------------------------------------

            elif op == "remove":

                current = getattr(
                    self.state,
                    field
                )

                if (
                    isinstance(current, list)
                    and value in current
                ):

                    current.remove(value)

            # -----------------------------------------
            # Clear
            # -----------------------------------------

            elif op == "clear":

                current = getattr(
                    self.state,
                    field
                )

                if isinstance(
                    current,
                    list
                ):

                    setattr(
                        self.state,
                        field,
                        []
                    )

                else:

                    setattr(
                        self.state,
                        field,
                        None
                    )

    # -----------------------------------------
    # Access state
    # -----------------------------------------

    def get_state(self):

        return self.state.to_dict()