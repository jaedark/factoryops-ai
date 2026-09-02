from typing import Protocol

from backend.app.schemas.agent import (
    ConversationMemory,
    MemoryMessage,
)


class MemoryStore(Protocol):
    def get_messages(
        self,
        session_id: str,
    ) -> list[MemoryMessage]: ...

    def append_message(
        self,
        session_id: str,
        message: MemoryMessage,
    ) -> None: ...

    def clear(
        self,
        session_id: str,
    ) -> None: ...


class MemoryStoreError(RuntimeError):
    """Session memory access error."""


class InMemoryMemoryStore:
    def __init__(
        self,
    ) -> None:
        self._sessions: dict[str, ConversationMemory] = {}

    def get_messages(
        self,
        session_id: str,
    ) -> list[MemoryMessage]:
        memory = self._sessions.get(
            session_id,
            ConversationMemory(
                session_id=session_id,
                messages=[],
            ),
        )
        return list(memory.messages)

    def append_message(
        self,
        session_id: str,
        message: MemoryMessage,
    ) -> None:
        memory = self._sessions.setdefault(
            session_id,
            ConversationMemory(
                session_id=session_id,
                messages=[],
            ),
        )
        memory.messages.append(message)

    def clear(
        self,
        session_id: str,
    ) -> None:
        self._sessions.pop(session_id, None)
