
import logging
from typing import Any

from openai import OpenAI

from backend.core.config import settings
from backend.core.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    RAG_NO_CONTEXT_MESSAGE,
    RAG_NO_DOCUMENTS_MESSAGE,
    QUERY_REWRITE_PROMPT,
)
from backend.ingestion.embeddings import embedding_generator
from backend.vectorstore.chroma_store import vector_store


logger = logging.getLogger(__name__)


class RAGAgent:

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Initialize OpenAI client lazily."""

        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not configured.")

            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_MAX_RETRIES,
            )

        return self._client

    def retrieve_context(self, question: str) -> list[dict[str, Any]]:
        """Generate query embedding and retrieve relevant chunks."""

        query_embedding = (
            embedding_generator.generate_query_embedding(question)
        )

        matches = vector_store.search(
            query_embedding=query_embedding,
            top_k=settings.TOP_K,
        )

        # Filter chunks using the configured similarity threshold.
        relevant_chunks = [
            match
            for match in matches
            if match["similarity"] >= settings.MIN_SIMILARITY
        ]

        return relevant_chunks

    @staticmethod
    def format_history(
        history: list[dict[str, str]] | None = None
    ) -> str:
        """Format recent conversation history."""

        if not history:
            return "No previous conversation."

        recent_history = history[-(settings.MAX_HISTORY_TURNS * 2):]

        formatted = []

        for message in recent_history:
            role = message.get("role", "user")
            content = message.get("content", "")

            if content:
                formatted.append(
                    f"{role.capitalize()}: {content}"
                )

        return "\n".join(formatted) or "No previous conversation."


    def contextualize_question(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Rewrite follow-up questions using conversation history."""

        if not history:
            return question

        formatted_history = self.format_history(history)

        prompt = QUERY_REWRITE_PROMPT.format(
            history=formatted_history,
            question=question,
        )

        response = self.client.responses.create(
            model=settings.LLM_MODEL,
            instructions=(
                "Rewrite the question for document retrieval. "
                "Return only the standalone question."
            ),
            input=prompt,
        )

        rewritten_question = response.output_text.strip()

        if not rewritten_question:
            logger.warning(
                "Query contextualization returned empty output."
            )
            return question

        logger.info(
            "Original question: %s | Rewritten question: %s",
            question,
            rewritten_question,
        )

        return rewritten_question

    @staticmethod
    def format_context(
        chunks: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Format retrieved chunks and preserve every citation mapping."""

        context_parts = []
        sources = []

        total_chars = 0

        for index, chunk in enumerate(chunks, start=1):

            metadata = chunk["metadata"]

            source = metadata.get("source", "Unknown document")
            page = metadata.get("page", "Unknown")

            excerpt = (
                f"[{index}] "
                f"Source: {source}, Page: {page}\n"
                f"{chunk['text']}"
            )

            if total_chars + len(excerpt) > settings.MAX_CONTEXT_CHARS:
                break

            context_parts.append(excerpt)
            total_chars += len(excerpt)

            # Maintain one source entry for every excerpt.
            sources.append({
                "citation": f"[{index}]",
                "filename": source,
                "page": page,
            })

        return "\n\n".join(context_parts), sources


    def generate_answer(
        self,
        question: str,
        context: str,
        history: str,
    ) -> str:
        """Generate an answer grounded in retrieved context."""

        user_prompt = RAG_USER_TEMPLATE.format(
            history=history,
            context=context,
            question=question,
        )

        response = self.client.responses.create(
            model=settings.LLM_MODEL,
            instructions=RAG_SYSTEM_PROMPT,
            input=user_prompt,
        )

        answer = response.output_text.strip()

        if not answer:
            raise RuntimeError("The LLM returned an empty response.")

        return answer

    def answer_question(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Complete RAG pipeline: retrieve, filter and generate."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1: Check document collection.
        if vector_store.count() == 0:
            return {
                "answer": RAG_NO_DOCUMENTS_MESSAGE,
                "sources": [],
                "retrieved_chunks": 0,
            }

        # Step 2: Resolve follow-up references using conversation history.
        contextualized_question = self.contextualize_question(
            question=question,
            history=history,
        )

        # Step 3: Retrieve using the standalone question.
        chunks = self.retrieve_context(contextualized_question)
        if not chunks:
            return {
                "answer": RAG_NO_CONTEXT_MESSAGE,
                "sources": [],
                "retrieved_chunks": 0,
            }

        # Step 4: Prepare numbered context and citations.
        context, sources = self.format_context(chunks)

        if not context:
            return {
                "answer": RAG_NO_CONTEXT_MESSAGE,
                "sources": [],
                "retrieved_chunks": 0,
            }

        # Step 5: Format conversation history.
        formatted_history = self.format_history(history)

        # Step 6: Generate grounded answer.
        answer = self.generate_answer(
            question=question,
            context=context,
            history=formatted_history,
        )

        logger.info(
            "RAG answer generated. Retrieved chunks: %d",
            len(chunks),
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(chunks),
        }


rag_agent = RAGAgent()
