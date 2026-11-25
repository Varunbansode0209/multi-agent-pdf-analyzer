
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import backend.agents.orchestrator as orchestrator_module
from backend.agents.orchestrator import Orchestrator
from backend.core.config import settings


def create_orchestrator():
    """Create an orchestrator without invoking its real constructor."""
    instance = Orchestrator.__new__(Orchestrator)
    instance.planner = Mock(name="test_planner")
    return instance


# Initialization tests

def test_orchestrator_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Orchestrator()


def test_orchestrator_initializes_planner(monkeypatch):
    agent_mock = Mock(name="planner")
    agent_factory = Mock(return_value=agent_mock)
    key_setter = Mock()

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-api-key")
    monkeypatch.setattr(orchestrator_module, "Agent", agent_factory)
    monkeypatch.setattr(
        orchestrator_module,
        "set_default_openai_key",
        key_setter,
    )

    instance = Orchestrator()

    key_setter.assert_called_once_with("test-api-key")
    agent_factory.assert_called_once()

    assert instance.planner is agent_mock

    agent_args = agent_factory.call_args.kwargs
    assert agent_args["name"] == "PDF Planner"
    assert len(agent_args["tools"]) == 2


# History tests

def test_format_history_handles_empty_history():
    assert Orchestrator.format_history(None) == ""
    assert Orchestrator.format_history([]) == ""


def test_format_history_formats_messages():
    history = [
        {"role": "user", "content": "Summarize the PDF"},
        {"role": "assistant", "content": "Here is the summary"},
    ]

    result = Orchestrator.format_history(history)

    assert "Previous conversation:" in result
    assert "User: Summarize the PDF" in result
    assert "Assistant: Here is the summary" in result


def test_format_history_ignores_invalid_messages():
    history = [
        {"role": "system", "content": "Ignore this"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Valid response"},
    ]

    result = Orchestrator.format_history(history)

    assert "Ignore this" not in result
    assert "Valid response" in result


def test_format_history_respects_history_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_HISTORY_TURNS", 1)

    history = [
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
    ]

    result = Orchestrator.format_history(history)

    assert "Old question" not in result
    assert "Recent question" in result
    assert "Recent answer" in result


# Message processing tests

def test_process_message_rejects_empty_message():
    instance = create_orchestrator()

    with pytest.raises(ValueError, match="Message cannot be empty"):
        asyncio.run(instance.process_message("   "))


def test_process_message_routes_to_conversation(monkeypatch):
    instance = create_orchestrator()

    runner_mock = AsyncMock(
        return_value=SimpleNamespace(final_output="Hello! How can I help?")
    )
    monkeypatch.setattr(orchestrator_module.Runner, "run", runner_mock)

    result = asyncio.run(instance.process_message("Hello"))

    assert result["route"] == "conversation"
    assert result["answer"] == "Hello! How can I help?"
    assert result["sources"] == []
    assert result["retrieved_chunks"] == 0

    runner_mock.assert_awaited_once()

    assert runner_mock.await_args.kwargs["input"] == "Hello"
    assert runner_mock.await_args.kwargs["max_turns"] == 4


def test_process_message_routes_to_rag(monkeypatch):
    instance = create_orchestrator()

    async def fake_runner(agent, input, context, max_turns):
        context.selected_tool = "answer_question"
        context.tool_result = {
            "answer": "The planner agent coordinates the workflow.",
            "sources": [{"source": "test.pdf", "page": 1}],
            "retrieved_chunks": 1,
        }

        return SimpleNamespace(final_output="RAG tool executed")

    runner_mock = AsyncMock(side_effect=fake_runner)
    monkeypatch.setattr(orchestrator_module.Runner, "run", runner_mock)

    result = asyncio.run(
        instance.process_message("What does the planner agent do?")
    )

    assert result["route"] == "answer_question"
    assert "planner agent" in result["answer"].lower()
    assert result["retrieved_chunks"] == 1
    assert result["planner_output"] == "RAG tool executed"


def test_process_message_routes_to_summarization(monkeypatch):
    instance = create_orchestrator()

    async def fake_runner(agent, input, context, max_turns):
        context.selected_tool = "summarize_documents"
        context.tool_result = {
            "message": "Summarization completed successfully.",
            "document_count": 1,
            "summaries": [{"filename": "test.pdf", "summary": "Test summary"}],
            "overall_summary": "Overall test summary",
        }

        return SimpleNamespace(final_output="Summarization tool executed")

    runner_mock = AsyncMock(side_effect=fake_runner)
    monkeypatch.setattr(orchestrator_module.Runner, "run", runner_mock)

    result = asyncio.run(
        instance.process_message("Summarize the uploaded documents")
    )

    assert result["route"] == "summarize_documents"
    assert result["document_count"] == 1
    assert result["overall_summary"] == "Overall test summary"


def test_process_message_includes_conversation_history(monkeypatch):
    instance = create_orchestrator()

    runner_mock = AsyncMock(
        return_value=SimpleNamespace(final_output="The second agent handles summaries.")
    )
    monkeypatch.setattr(orchestrator_module.Runner, "run", runner_mock)

    history = [
        {"role": "user", "content": "Which agents are available?"},
        {"role": "assistant", "content": "There are two agents."},
    ]

    asyncio.run(
        instance.process_message(
            "What does the second one do?",
            history=history,
        )
    )

    planner_input = runner_mock.await_args.kwargs["input"]

    assert "Previous conversation:" in planner_input
    assert "Which agents are available?" in planner_input
    assert "What does the second one do?" in planner_input


def test_process_message_passes_history_into_context(monkeypatch):
    instance = create_orchestrator()
    captured_context = {}

    async def fake_runner(agent, input, context, max_turns):
        captured_context["history"] = context.history
        return SimpleNamespace(final_output="Response")

    runner_mock = AsyncMock(side_effect=fake_runner)
    monkeypatch.setattr(orchestrator_module.Runner, "run", runner_mock)

    history = [
        {"role": "user", "content": "Previous question"},
    ]

    asyncio.run(
        instance.process_message("Follow-up question", history=history)
    )

    assert captured_context["history"] == history
