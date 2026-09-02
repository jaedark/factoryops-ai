from backend.app.schemas.agent import MemoryMessage


class ContextBuilder:
    @staticmethod
    def build(
        current_message: str,
        memory_messages: list[MemoryMessage],
        max_messages: int = 6,
    ) -> str:
        recent_messages = memory_messages[-max_messages:]

        if not recent_messages:
            return current_message

        previous_conversation = "\n".join(
            (
                f"{ContextBuilder._format_role(message.role)}: "
                f"{message.content}"
            )
            for message in recent_messages
        )

        return (
            "Previous Conversation:\n"
            f"{previous_conversation}\n\n"
            "Current Request:\n"
            f"{current_message}"
        )

    @staticmethod
    def _format_role(
        role: str,
    ) -> str:
        if role == "user":
            return "User"
        if role == "assistant":
            return "Assistant"
        return role.capitalize()
