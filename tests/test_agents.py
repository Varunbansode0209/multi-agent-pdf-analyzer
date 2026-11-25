
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import backend.agents.rag_agent as rag_module
import backend.agents.summarization_agent as summary_module

from backend.agents.rag_agent import RAGAgent
from backend.agents.summarization_agent import SummarizationAgent
from backend.core.config import settings


# ==================================================
# SHARED TEST DATA
# ==================================================

def sample_chunk(
    text="Sample document content",
    source="test.pdf",
    page=1,
    chunk_index=0,
):
    return {
        "id": f"chunk-{page}-{chunk_index}",
        "text": text,
        "metadata": {
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
        },
        "similarity": 0.9,
        "distance": 0.1,
    }


def fake_response(text):
    return SimpleNamespace(output_text=text)


# ==================================================
# RAG AGENT TESTS
# ==================================================

def test_rag_agent_initializes_without_client():
    agent = RAGAgent()

    assert agent._client is None


def test_format_history_handles_empty_history():
    result = RAGAgent.format_history(None)

    assert result == "No previous conversation."


def test_format_history_formats_messages():
    history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG retrieves context."},
    ]

    result = RAGAgent.format_history(history)

    assert "User: What is RAG?" in result
    assert "Assistant: RAG retrieves context." in result


def test_format_history_respects_history_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_HISTORY_TURNS", 1)

    history = [
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
    ]

    result = RAGAgent.format_history(history)

    assert "Old question" not in result
    assert "Recent question" in result


def test_contextualize_question_without_history():
    agent = RAGAgent()

    question = "What is machine learning?"

    result = agent.contextualize_question(question)

    assert result == question
    assert agent._client is None


def test_contextualize_question_rewrites_follow_up():
    agent = RAGAgent()

    agent._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=Mock(
                return_value=fake_response(
                    "What is the responsibility of the Summarization Agent?"
                )
            )
        )
    )

    history = [
        {"role": "user", "content": "Which agents are required?"},
        {"role": "assistant", "content": "RAG and Summarization."},
    ]

    result = agent.contextualize_question(
        "What does the second one do?",
        history,
    )

    assert "Summarization Agent" in result


def test_contextualize_question_falls_back_on_empty_response():
    agent = RAGAgent()

    agent._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=Mock(return_value=fake_response(""))
        )
    )

    question = "What does the second agent do?"

    result = agent.contextualize_question(
        question,
        [{"role": "user", "content": "Name the agents."}],
    )

    assert result == question


def test_format_context_preserves_citation_mapping():
    chunks = [
        sample_chunk("First content", "first.pdf", 1),
        sample_chunk("Second content", "second.pdf", 3),
    ]

    context, sources = RAGAgent.format_context(chunks)

    assert "[1]" in context
    assert "[2]" in context

    assert sources == [
        {
            "citation": "[1]",
            "filename": "first.pdf",
            "page": 1,
        },
        {
            "citation": "[2]",
            "filename": "second.pdf",
            "page": 3,
        },
    ]


def test_format_context_respects_character_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 100)

    chunks = [
        sample_chunk("A" * 60),
        sample_chunk("B" * 60, page=2),
    ]

    context, sources = RAGAgent.format_context(chunks)

    assert len(sources) == 1
    assert "A" * 60 in context
    assert "B" * 60 not in context


def test_retrieve_context_filters_low_similarity(monkeypatch):
    agent = RAGAgent()

    monkeypatch.setattr(
        settings,
        "MIN_SIMILARITY",
        0.5,
    )

    monkeypatch.setattr(
        rag_module.embedding_generator,
        "generate_query_embedding",
        Mock(return_value=[1.0, 0.0]),
    )

    monkeypatch.setattr(
        rag_module.vector_store,
        "search",
        Mock(return_value=[
            sample_chunk("Relevant content"),
            {
                **sample_chunk("Irrelevant content"),
                "similarity": 0.2,
            },
        ]),
    )

    results = agent.retrieve_context("What is RAG?")

    assert len(results) == 1
    assert results[0]["text"] == "Relevant content"


def test_generate_answer_returns_llm_response():
    agent = RAGAgent()

    agent._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=Mock(
                return_value=fake_response("RAG retrieves relevant context.")
            )
        )
    )

    result = agent.generate_answer(
        question="What is RAG?",
        context="RAG retrieves relevant documents.",
        history="No previous conversation.",
    )

    assert result == "RAG retrieves relevant context."


def test_generate_answer_rejects_empty_response():
    agent = RAGAgent()

    agent._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=Mock(return_value=fake_response(""))
        )
    )

    with pytest.raises(RuntimeError, match="empty response"):
        agent.generate_answer(
            question="What is RAG?",
            context="Some context",
            history="",
        )


def test_answer_question_rejects_empty_question():
    agent = RAGAgent()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        agent.answer_question("   ")


def test_answer_question_handles_empty_collection(monkeypatch):
    agent = RAGAgent()

    monkeypatch.setattr(
        rag_module.vector_store,
        "count",
        Mock(return_value=0),
    )

    result = agent.answer_question("What is RAG?")

    assert result["retrieved_chunks"] == 0
    assert result["sources"] == []
    assert result["answer"] == rag_module.RAG_NO_DOCUMENTS_MESSAGE


def test_answer_question_handles_no_relevant_context(monkeypatch):
    agent = RAGAgent()

    monkeypatch.setattr(
        rag_module.vector_store,
        "count",
        Mock(return_value=5),
    )

    monkeypatch.setattr(
        agent,
        "retrieve_context",
        Mock(return_value=[]),
    )

    result = agent.answer_question("What is RAG?")

    assert result["retrieved_chunks"] == 0
    assert result["sources"] == []
    assert result["answer"] == rag_module.RAG_NO_CONTEXT_MESSAGE


def test_answer_question_returns_grounded_answer(monkeypatch):
    agent = RAGAgent()

    chunks = [
        sample_chunk("RAG retrieves relevant document sections.")
    ]

    monkeypatch.setattr(
        rag_module.vector_store,
        "count",
        Mock(return_value=1),
    )

    monkeypatch.setattr(
        agent,
        "retrieve_context",
        Mock(return_value=chunks),
    )

    monkeypatch.setattr(
        agent,
        "generate_answer",
        Mock(return_value="RAG retrieves relevant sections. [1]"),
    )

    result = agent.answer_question("What is RAG?")

    assert result["answer"] == "RAG retrieves relevant sections. [1]"
    assert result["retrieved_chunks"] == 1
    assert result["sources"][0]["citation"] == "[1]"


def test_answer_question_passes_history_to_contextualization(monkeypatch):
    agent = RAGAgent()

    history = [
        {"role": "user", "content": "Which agents are required?"},
        {"role": "assistant", "content": "RAG and Summarization."},
    ]

    monkeypatch.setattr(
        rag_module.vector_store,
        "count",
        Mock(return_value=1),
    )

    contextualize = Mock(return_value="What does the Summarization Agent do?")

    monkeypatch.setattr(
        agent,
        "contextualize_question",
        contextualize,
    )

    monkeypatch.setattr(
        agent,
        "retrieve_context",
        Mock(return_value=[sample_chunk()]),
    )

    monkeypatch.setattr(
        agent,
        "generate_answer",
        Mock(return_value="It summarizes documents."),
    )

    agent.answer_question(
        "What does the second one do?",
        history,
    )

    contextualize.assert_called_once_with(
        question="What does the second one do?",
        history=history,
    )


# ==================================================
# SUMMARIZATION AGENT TESTS
# ==================================================

def test_summarization_agent_initializes_without_client():
    agent = SummarizationAgent()

    assert agent._client is None


def test_generate_returns_summary():
    agent = SummarizationAgent()

    agent._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=fake_response("Generated summary.")
            )
        )
    )

    result = asyncio.run(
        agent._generate(
            prompt="Summarize this document.",
            instructions="Be factual.",
        )
    )

    assert result == "Generated summary."


def test_generate_rejects_empty_summary():
    agent = SummarizationAgent()

    agent._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(return_value=fake_response(""))
        )
    )

    with pytest.raises(RuntimeError, match="empty summary"):
        asyncio.run(
            agent._generate(
                prompt="Summarize this document.",
                instructions="Be factual.",
            )
        )


def test_group_chunks_separates_documents():
    agent = SummarizationAgent()

    chunks = [
        sample_chunk("First document", "first.pdf"),
        sample_chunk("Second document", "second.pdf"),
        sample_chunk("More first document", "first.pdf", page=2),
    ]

    result = agent.group_chunks(chunks)

    assert set(result.keys()) == {"first.pdf", "second.pdf"}
    assert len(result["first.pdf"][0]) == 2
    assert len(result["second.pdf"][0]) == 1


def test_group_chunks_respects_batch_character_limit(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(
        settings,
        "MAP_BATCH_CHARS",
        10,
    )

    chunks = [
        sample_chunk("A" * 6),
        sample_chunk("B" * 6, page=2),
    ]

    result = agent.group_chunks(chunks)

    assert len(result["test.pdf"]) == 2


def test_prepare_map_prompt_includes_source_and_pages():
    agent = SummarizationAgent()

    batch = [
        sample_chunk("First page content", page=1),
        sample_chunk("Second page content", page=2),
    ]

    prompt = agent.prepare_map_prompt(
        source="test.pdf",
        batch=batch,
    )

    assert "test.pdf" in prompt
    assert "pages 1-2" in prompt
    assert "First page content" in prompt
    assert "Second page content" in prompt


def test_map_batch_generates_summary():
    agent = SummarizationAgent()

    agent._generate = AsyncMock(return_value="Mapped summary.")

    batch = [sample_chunk("Document content")]

    result = asyncio.run(
        agent.map_batch(
            source="test.pdf",
            batch=batch,
            semaphore=asyncio.Semaphore(1),
        )
    )

    assert result == "Mapped summary."
    agent._generate.assert_awaited_once()


def test_map_documents_collects_batch_summaries():
    agent = SummarizationAgent()

    agent.map_batch = AsyncMock(
        side_effect=["Summary one", "Summary two"]
    )

    grouped = {
        "test.pdf": [
            [sample_chunk("First batch")],
            [sample_chunk("Second batch", page=2)],
        ]
    }

    result = asyncio.run(agent.map_documents(grouped))

    assert result == {
        "test.pdf": [
            "Summary one",
            "Summary two",
        ]
    }


def test_map_documents_raises_when_batch_fails():
    agent = SummarizationAgent()

    agent.map_batch = AsyncMock(
        side_effect=RuntimeError("Simulated LLM failure")
    )

    grouped = {
        "test.pdf": [
            [sample_chunk("First batch")],
        ]
    }

    with pytest.raises(RuntimeError, match="Some document batches failed"):
        asyncio.run(agent.map_documents(grouped))


def test_reduce_summary_combines_intermediate_summaries(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(
        settings,
        "REDUCE_MAX_CHARS",
        1000,
    )

    agent._generate = AsyncMock(return_value="Final summary.")

    result = asyncio.run(
        agent.reduce_summary([
            "First section summary.",
            "Second section summary.",
        ])
    )

    assert result == "Final summary."
    agent._generate.assert_awaited_once()


def test_reduce_summary_reduces_oversized_input(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(
        settings,
        "REDUCE_MAX_CHARS",
        10,
    )

    agent._generate = AsyncMock(
        side_effect=["x", "x", "x", "Final summary."]
    )

    result = asyncio.run(
        agent.reduce_summary([
            "12345678",
            "abcdefgh",
            "ABCDEFGH",
        ])
    )

    assert result == "Final summary."
    assert agent._generate.await_count == 4


def test_reduce_summary_rejects_empty_input():
    agent = SummarizationAgent()

    with pytest.raises(ValueError, match="No intermediate summaries"):
        asyncio.run(agent.reduce_summary([]))


def test_summarize_documents_handles_empty_collection(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(
        summary_module.vector_store,
        "get_all_chunks",
        Mock(return_value=[]),
    )

    result = asyncio.run(agent.summarize_documents())

    assert result["summaries"] == []
    assert result["message"] == summary_module.SUMMARY_NO_DOCUMENTS_MESSAGE


def test_summarize_documents_handles_single_document(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(
        summary_module.vector_store,
        "get_all_chunks",
        Mock(return_value=[sample_chunk()]),
    )

    agent.map_documents = AsyncMock(
        return_value={"test.pdf": ["Mapped summary."]}
    )

    agent.summarize_document = AsyncMock(
        return_value={
            "filename": "test.pdf",
            "summary": "Final document summary.",
        }
    )

    result = asyncio.run(agent.summarize_documents())

    assert result["document_count"] == 1
    assert result["summaries"][0]["filename"] == "test.pdf"
    assert result["overall_summary"] == "Final document summary."


def test_summarize_documents_handles_multiple_documents(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(
        summary_module.vector_store,
        "get_all_chunks",
        Mock(return_value=[
            sample_chunk("First document", "first.pdf"),
            sample_chunk("Second document", "second.pdf"),
        ]),
    )

    agent.map_documents = AsyncMock(
        return_value={
            "first.pdf": ["First mapped summary."],
            "second.pdf": ["Second mapped summary."],
        }
    )

    agent.summarize_document = AsyncMock(
        side_effect=[
            {
                "filename": "first.pdf",
                "summary": "First final summary.",
            },
            {
                "filename": "second.pdf",
                "summary": "Second final summary.",
            },
        ]
    )

    agent._generate = AsyncMock(
        return_value="Combined summary."
    )

    result = asyncio.run(agent.summarize_documents())

    assert result["document_count"] == 2
    assert len(result["summaries"]) == 2
    assert result["overall_summary"] == "Combined summary."

    agent._generate.assert_awaited_once()


def test_reduce_summary_respects_concurrency_limit(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(settings, "REDUCE_MAX_CHARS", 10)
    monkeypatch.setattr(settings, "MAP_CONCURRENCY", 2)

    active_calls = 0
    max_active_calls = 0

    async def fake_generate(prompt, instructions):
        nonlocal active_calls, max_active_calls

        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)

        try:
            await asyncio.sleep(0.05)

            if "Produce a coherent, factual document summary." in instructions:
                return "Final summary."

            return "x"
        finally:
            active_calls -= 1

    agent._generate = fake_generate

    result = asyncio.run(
        agent.reduce_summary([
            "12345678",
            "abcdefgh",
            "ABCDEFGH",
        ])
    )

    assert result == "Final summary."
    assert max_active_calls <= 2

def test_reduce_multiple_documents_handles_oversized_input(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(settings, "REDUCE_MAX_CHARS", 100)
    monkeypatch.setattr(settings, "MAP_CONCURRENCY", 2)

    calls = []

    async def fake_generate(prompt, instructions):
        calls.append(prompt)

        if "Write an overall summary:" in prompt:
            return "Overall summary."

        return "Compressed document summaries."

    agent._generate = fake_generate

    semaphore = asyncio.Semaphore(2)

    result = asyncio.run(
        agent.reduce_multiple_documents(
            document_summaries=[
                {
                    "filename": "report_a.pdf",
                    "summary": "A" * 80,
                },
                {
                    "filename": "report_b.pdf",
                    "summary": "B" * 80,
                },
                {
                    "filename": "report_c.pdf",
                    "summary": "C" * 80,
                },
            ],
            semaphore=semaphore,
        )
    )

    assert result == "Overall summary."
    assert len(calls) >= 2

    intermediate_prompts = calls[:-1]

    for filename in[
        "report_a.pdf",
        "report_b.pdf",
        "report_c.pdf"
    ]:
        assert any(
            filename in prompt
            for prompt in intermediate_prompts
        )
    assert "Write an overall summary:" in calls[-1]

def test_reduce_summary_handles_single_oversized_input(monkeypatch):
    agent = SummarizationAgent()

    monkeypatch.setattr(settings, "REDUCE_MAX_CHARS", 100)

    calls = []

    async def fake_generate(prompt, instructions):
        calls.append(prompt)

        if "Write the final summary with this structure:" in prompt:
            return "Final summary."

        return "Compressed summary."

    agent._generate = fake_generate

    result = asyncio.run(
        agent.reduce_summary(
            summaries=["A" * 200],
        )
    )

    assert result == "Final summary."
    assert len(calls) == 2
    assert len(calls[0]) > 100

def test_group_chunks_rejects_oversized_individual_chunk(monkeypatch):
    monkeypatch.setattr(settings, "MAP_BATCH_CHARS", 100)

    chunks = [
        sample_chunk(
            text="A" * 101,
            source="large.pdf",
        )
    ]

    with pytest.raises(
        ValueError,
        match="exceeds MAP_BATCH_CHARS",
    ):
        SummarizationAgent.group_chunks(chunks)