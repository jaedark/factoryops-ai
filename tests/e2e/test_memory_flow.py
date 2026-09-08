from unittest.mock import patch

import pytest

from backend.app.services.llm_service import LlmService
from tests.e2e.helpers import build_text_response, build_tool_response


pytestmark = pytest.mark.e2e


def _first_content_text(mock_generate, call_index: int) -> str:
    contents = mock_generate.call_args_list[call_index].kwargs["contents"]
    return contents[0].parts[0].text


def test_e2e_06_session_memory_is_injected_into_follow_up(
    e2e_environment,
):
    session_id = "e2e-memory-robot"
    first_message = "Robot-01 현재 상태를 확인해줘."
    first_answer = "Robot-01 현재 상태를 확인했습니다."
    follow_up = "그 설비의 과거 장애도 찾아줘."
    with patch.object(
        LlmService,
        "generate_content",
        side_effect=[
            build_tool_response(
                "get_equipment_status",
                {"equipment_id": "Robot-01"},
            ),
            build_text_response(first_answer),
            build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
            build_text_response("Robot-01 과거 장애 두 건을 확인했습니다."),
        ],
    ) as mock_generate:
        first = e2e_environment.client.post(
            "/agent/chat",
            json={"message": first_message, "session_id": session_id},
        )
        second = e2e_environment.client.post(
            "/agent/chat",
            json={"message": follow_up, "session_id": session_id},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["steps"][0]["tool_called"] == (
        "get_equipment_incidents"
    )
    context = _first_content_text(mock_generate, 2)
    assert "Previous Conversation:" in context
    assert first_message in context
    assert first_answer in context
    assert "Current Request:" in context
    assert follow_up in context
    assert "Robot-01" in context
    messages = e2e_environment.memory_store.get_messages(session_id)
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_e2e_07_stateless_follow_up_has_no_previous_context(
    e2e_environment,
):
    first_message = "Robot-01 현재 상태를 확인해줘."
    follow_up = "그 설비의 과거 장애도 찾아줘."
    with patch.object(
        LlmService,
        "generate_content",
        side_effect=[
            build_text_response("Robot-01 상태 확인 완료"),
            build_text_response("대상을 특정할 수 없습니다."),
        ],
    ) as mock_generate:
        first = e2e_environment.client.post(
            "/agent/chat",
            json={"message": first_message},
        )
        second = e2e_environment.client.post(
            "/agent/chat",
            json={"message": follow_up},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert _first_content_text(mock_generate, 1) == follow_up
    assert "Previous Conversation:" not in _first_content_text(
        mock_generate,
        1,
    )
