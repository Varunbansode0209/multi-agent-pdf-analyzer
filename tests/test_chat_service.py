
import asyncio
from unittest.mock import AsyncMock

import pytest

import backend.services.chat_service as chat_module
from backend.services.chat_service import ChatService
from backend.core.config import settings


@pytest.fixture
def service():
    return ChatService()


def test_process_message_rejects_empty_message(service):
    with pytest.raises(ValueError, match="Message cannot be empty"):
        asyncio.run(service.process_message(" ", "session-1"))


def test_process_message_rejects_empty_session_id(service):
    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        asyncio.run(service.process_message("Hello", " "))


def test_process_message_stores_conversation(service, monkeypatch):
    mock_process = AsyncMock(
        return_value={
            "answer": "The document discusses AI.",
            "route": "answer_question",
        }
    )
    monkeypatch.setattr(
        chat_module.orchestrator,
        "process_message",
        mock_process,
    )

    result = asyncio.run(
        service.process_message("What is this document about?", "session-1")
    )

    assert result["session_id"] == "session-1"
    assert result["answer"] == "The document discusses AI."

    assert service.conversations["session-1"] == [
        {
            "role": "user",
            "content": "What is this document about?",
        },
        {
            "role": "assistant",
            "content": "The document discusses AI.",
        },
    ]


def test_process_message_strips_user_message(service, monkeypatch):
    monkeypatch.setattr(
        chat_module.orchestrator,
        "process_message",
        AsyncMock(return_value={"answer": "Hello!"}),
    )

    asyncio.run(
        service.process_message("  Hello  ", "session-1")
    )

    assert service.conversations["session-1"][0]["content"] == "Hello"



def test_process_message_passes_existing_history(service, monkeypatch):
    existing_history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    service.conversations["session-1"] = existing_history.copy()

    captured = {}

    async def fake_process(message, history):
        captured["message"] = message
        captured["history"] = history.copy()
        return {"answer": "Follow-up answer"}

    monkeypatch.setattr(
        chat_module.orchestrator,
        "process_message",
        fake_process,
    )

    asyncio.run(
        service.process_message("Follow-up question", "session-1")
    )

    assert captured["message"] == "Follow-up question"
    assert captured["history"] == existing_history



def test_sessions_are_isolated(service, monkeypatch):
    captured_histories = []

    async def fake_process(message, history):
        captured_histories.append(history.copy())
        return {"answer": "Response"}

    monkeypatch.setattr(
        chat_module.orchestrator,
        "process_message",
        fake_process,
    )

    asyncio.run(service.process_message("Question 1", "session-1"))
    asyncio.run(service.process_message("Question 2", "session-2"))

    assert len(service.conversations["session-1"]) == 2
    assert len(service.conversations["session-2"]) == 2

    assert captured_histories[0] == []
    assert captured_histories[1] == []



def test_process_message_respects_history_limit(service, monkeypatch):
    monkeypatch.setattr(settings, "MAX_HISTORY_TURNS", 2)

    mock_process = AsyncMock(
        return_value={"answer": "Response"}
    )
    monkeypatch.setattr(
        chat_module.orchestrator,
        "process_message",
        mock_process,
    )

    for index in range(4):
        asyncio.run(
            service.process_message(
                f"Question {index}",
                "session-1",
            )
        )

    history = service.conversations["session-1"]

    assert len(history) == 4
    assert history[0]["content"] == "Question 2"
    assert history[-1]["content"] == "Response"


def test_process_message_uses_summary_as_assistant_history(service, monkeypatch):
    mock_process = AsyncMock(
        return_value={
            "message": "Summarization completed.",
            "overall_summary": "This document explains machine learning.",
            "route": "summarize_documents",
        }
    )
    monkeypatch.setattr(
        chat_module.orchestrator,
        "process_message",
        mock_process,
    )

    asyncio.run(
        service.process_message("Summarize this document", "session-1")
    )

    history = service.conversations["session-1"]

    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == (
        "This document explains machine learning."
    )


def test_failed_processing_does_not_store_history(service, monkeypatch):
    mock_process = AsyncMock(
        side_effect=RuntimeError("LLM API failed")
    )
    monkeypatch.setattr(
        chat_module.orchestrator,
        "process_message",
        mock_process,
    )

    with pytest.raises(RuntimeError, match="LLM API failed"):
        asyncio.run(
            service.process_message("Explain this PDF", "session-1")
        )

    assert "session-1" not in service.conversations


def test_clear_conversation_removes_existing_session(service):
    service.conversations["session-1"] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    result = service.clear_conversation("session-1")

    assert result is True
    assert "session-1" not in service.conversations


def test_clear_conversation_returns_false_for_missing_session(service):
    result = service.clear_conversation("unknown-session")

    assert result is False
