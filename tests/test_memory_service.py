from backend.app.schemas.agent import MemoryMessage
from backend.app.services.context_builder import ContextBuilder
from backend.app.services.memory_service import (
    InMemoryMemoryStore,
)


def test_memory_store_returns_empty_messages_for_new_session():
    store = InMemoryMemoryStore()

    assert store.get_messages("demo-001") == []


def test_memory_store_appends_and_reads_messages():
    store = InMemoryMemoryStore()
    store.append_message(
        "demo-001",
        MemoryMessage(
            role="user",
            content="Robot-01 현재 상태 알려줘",
        ),
    )
    store.append_message(
        "demo-001",
        MemoryMessage(
            role="assistant",
            content="Robot-01은 현재 high risk입니다.",
        ),
    )

    messages = store.get_messages("demo-001")

    assert [message.role for message in messages] == [
        "user",
        "assistant",
    ]


def test_memory_store_keeps_sessions_isolated():
    store = InMemoryMemoryStore()
    store.append_message(
        "session-a",
        MemoryMessage(
            role="user",
            content="Robot-01 상태 알려줘",
        ),
    )
    store.append_message(
        "session-b",
        MemoryMessage(
            role="user",
            content="Conveyor-01 상태 알려줘",
        ),
    )

    session_a = store.get_messages("session-a")
    session_b = store.get_messages("session-b")

    assert session_a[0].content == "Robot-01 상태 알려줘"
    assert session_b[0].content == "Conveyor-01 상태 알려줘"


def test_memory_store_clears_session_messages():
    store = InMemoryMemoryStore()
    store.append_message(
        "demo-001",
        MemoryMessage(
            role="user",
            content="clear me",
        ),
    )

    store.clear("demo-001")

    assert store.get_messages("demo-001") == []


def test_context_builder_formats_previous_conversation_and_current_request():
    context = ContextBuilder.build(
        current_message="그럼 과거 장애는?",
        memory_messages=[
            MemoryMessage(
                role="user",
                content="Robot-01 현재 상태 알려줘",
            ),
            MemoryMessage(
                role="assistant",
                content="Robot-01은 현재 high risk입니다.",
            ),
        ],
    )

    assert "Previous Conversation:" in context
    assert "User: Robot-01 현재 상태 알려줘" in context
    assert "Assistant: Robot-01은 현재 high risk입니다." in context
    assert "Current Request:\n그럼 과거 장애는?" in context


def test_context_builder_applies_recent_message_limit():
    messages = [
        MemoryMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=f"message-{index}",
        )
        for index in range(8)
    ]

    context = ContextBuilder.build(
        current_message="follow-up",
        memory_messages=messages,
        max_messages=4,
    )

    assert "message-0" not in context
    assert "message-3" not in context
    assert "message-4" in context
    assert "message-7" in context
