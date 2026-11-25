
import asyncio
import logging
from collections import defaultdict
from typing import Any

from openai import AsyncOpenAI

from backend.core.config import settings
from backend.core.prompts import (
    MAP_PROMPT,
    REDUCE_PROMPT,
    INTERMEDIATE_REDUCE_PROMPT,
    MULTI_DOC_REDUCE_PROMPT,
    MULTI_DOC_INTERMEDIATE_REDUCE_PROMPT,
    SUMMARY_NO_DOCUMENTS_MESSAGE,
)
from backend.vectorstore.chroma_store import vector_store


logger = logging.getLogger(__name__)


class SummarizationAgent:

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Initialize OpenAI asynchronous client lazily."""

        if self._client is None:

            if not settings.OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured."
                )

            self._client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_MAX_RETRIES,
            )

        return self._client

    async def _generate(
        self,
        prompt: str,
        instructions: str,
    ) -> str:
        """Generate a summary using the configured LLM."""

        response = await self.client.responses.create(
            model=settings.LLM_MODEL,
            instructions=instructions,
            input=prompt,
        )

        result = response.output_text.strip()

        if not result:
            raise RuntimeError(
                "LLM returned an empty summary."
            )

        return result

    @staticmethod
    def group_chunks(
        chunks: list[dict[str, Any]],
    ) -> dict[str, list[list[dict[str, Any]]]]:
        """Group chunks by document and batch them by character limit."""

        documents = defaultdict(list)

        for chunk in chunks:
            source = chunk["metadata"]["source"]
            documents[source].append(chunk)

        grouped_documents = {}

        for source, document_chunks in documents.items():

            batches = []
            current_batch = []
            current_chars = 0

            for chunk in document_chunks:

                chunk_size = len(chunk["text"])

                if chunk_size > settings.MAP_BATCH_CHARS:
                    raise ValueError(
                        f"Chunk from '{source}' exceeds MAP_BATCH_CHARS "
                        f"({chunk_size} > {settings.MAP_BATCH_CHARS})."
                )
                if (
                    current_batch
                    and current_chars + chunk_size
                    > settings.MAP_BATCH_CHARS
                ):
                    batches.append(current_batch)
                    current_batch = []
                    current_chars = 0

                current_batch.append(chunk)
                current_chars += chunk_size

            if current_batch:
                batches.append(current_batch)

            grouped_documents[source] = batches

        return grouped_documents

    @staticmethod
    def prepare_map_prompt(
        source: str,
        batch: list[dict[str, Any]],
    ) -> str:
        """Prepare a map prompt for a batch of document chunks."""

        text = "\n\n".join(
            chunk["text"] for chunk in batch
        )

        pages = sorted({
            chunk["metadata"]["page"]
            for chunk in batch
        })

        page_range = (
            str(pages[0])
            if len(pages) == 1
            else f"{pages[0]}-{pages[-1]}"
        )

        return MAP_PROMPT.format(
            source=source,
            page_range=page_range,
            text=text,
        )

    async def map_batch(
        self,
        source: str,
        batch: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Summarize one document batch."""

        prompt = self.prepare_map_prompt(
            source=source,
            batch=batch,
        )

        async with semaphore:

            return await self._generate(
                prompt=prompt,
                instructions=(
                    "You are a factual document section summarizer."
                ),
            )

    async def map_documents(
        self,
        grouped_documents: dict[
            str, list[list[dict[str, Any]]]
        ],
    ) -> dict[str, list[str]]:
        """Summarize batches concurrently with controlled concurrency."""

        semaphore = asyncio.Semaphore(
            max(1, settings.MAP_CONCURRENCY)
        )

        tasks = []
        task_metadata = []

        for source, batches in grouped_documents.items():

            for batch_index, batch in enumerate(batches):

                tasks.append(
                    self.map_batch(
                        source=source,
                        batch=batch,
                        semaphore=semaphore,
                    )
                )

                task_metadata.append(
                    (source, batch_index)
                )

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        summaries = defaultdict(list)
        errors = []

        for metadata, result in zip(
            task_metadata,
            results,
        ):

            source, batch_index = metadata

            if isinstance(result, Exception):
                errors.append(
                    f"{source}, batch {batch_index + 1}: {result}"
                )
                continue

            summaries[source].append(result)

        if errors:
            raise RuntimeError(
                "Some document batches failed:\n"
                + "\n".join(errors)
            )

        return dict(summaries)


    async def reduce_summary(
        self,
        summaries: list[str],
        semaphore: asyncio.Semaphore | None = None,
    ) -> str:
        """Combine summaries while limiting concurrent LLM calls."""

        if not summaries:
            raise ValueError(
                "No intermediate summaries provided."
            )

        if semaphore is None:
            semaphore = asyncio.Semaphore(
            max(1, settings.MAP_CONCURRENCY)
        )

        async def generate_limited(
            prompt: str,
            instructions: str,
        ) -> str:
            async with semaphore:
                return await self._generate(
                    prompt=prompt,
                    instructions=instructions,
                )

        current_summaries = summaries

        while True:

            combined = "\n\n".join(
                current_summaries
            )

            if len(combined) <= settings.REDUCE_MAX_CHARS:
                break

            # Split intermediate summaries into smaller groups.
            groups = []
            current_group = []
            current_chars = 0

            for summary in current_summaries:

                if (
                    current_group
                    and current_chars + len(summary)
                    > settings.REDUCE_MAX_CHARS
                ):
                    groups.append(current_group)
                    current_group = []
                    current_chars = 0

                current_group.append(summary)
                current_chars += len(summary)

            if current_group:
                groups.append(current_group)

            # Prevent an infinite reduction loop.
            # Handle a single summary that exceeds the character limit.
            if (
                len(groups) == 1
                and len(groups[0]) == 1
                and len(groups[0][0]) > settings.REDUCE_MAX_CHARS
            ):
                reduced_summaries = await asyncio.gather(
                    generate_limited(
                        prompt=INTERMEDIATE_REDUCE_PROMPT.format(
                            summaries=groups[0][0]
                        ),
                        instructions=(
                            "Shorten this summary while preserving "
                            "all important facts."
                        ),
                    )
                )

                if len(reduced_summaries[0]) >= len(groups[0][0]):
                    raise RuntimeError(
                        "Unable to reduce oversized summary."
                    )

                current_summaries = reduced_summaries
                continue

            # Prevent an infinite reduction loop.
            if (
                len(groups) == 1
                and len(groups[0]) == len(current_summaries)
            ):
                raise RuntimeError(
                    "Unable to reduce intermediate summaries."
                )

            current_summaries = await asyncio.gather(
                *[
                    generate_limited(
                        prompt=INTERMEDIATE_REDUCE_PROMPT.format(
                            summaries="\n\n".join(group)
                        ),
                        instructions=(
                            "Merge summaries without losing important facts."
                        ),
                    )
                    for group in groups
                ]
            )

        return await generate_limited(
            prompt=REDUCE_PROMPT.format(
                summaries=combined
            ),
            instructions=(
                "Produce a coherent, factual document summary."
            ),
        )



    async def reduce_multiple_documents(
        self,
        document_summaries: list[dict[str, str]],
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Reduce multiple document summaries while preserving filenames."""

        if not document_summaries:
            raise ValueError(
                "No document summaries provided."
            )

        current_summaries = [
            f"Document: {item['filename']}\n{item['summary']}"
            for item in document_summaries
        ]

        async def generate_limited(
            prompt: str,
            instructions: str,
        ) -> str:
            async with semaphore:
                return await self._generate(
                    prompt=prompt,
                    instructions=instructions,
                )

        while True:

            combined = "\n\n".join(current_summaries)

            if len(combined) <= settings.REDUCE_MAX_CHARS:
                break

            groups = []
            current_group = []
            current_chars = 0

            for summary in current_summaries:

                if (
                    current_group
                    and current_chars + len(summary) + 2
                    > settings.REDUCE_MAX_CHARS
                ):
                    groups.append(current_group)
                    current_group = []
                    current_chars = 0

                current_group.append(summary)
                current_chars += len(summary) + 2

            if current_group:
                groups.append(current_group)

            tasks = []

            for group in groups:

                group_text = "\n\n".join(group)

                if len(group) == 1 and len(group_text) > settings.REDUCE_MAX_CHARS:
                    instructions = (
                        "Shorten this document summary while preserving "
                        "its exact filename and important facts."
                    )
                else:
                    instructions = (
                        "Compress these document summaries while "
                        "preserving every filename and its associated facts."
                    )

                tasks.append(
                    generate_limited(
                        prompt=MULTI_DOC_INTERMEDIATE_REDUCE_PROMPT.format(
                            summaries=group_text
                        ),
                        instructions=instructions,
                    )
                )

            next_summaries = await asyncio.gather(*tasks)

            # Ensure each reduction pass actually decreases input size.
            if sum(map(len, next_summaries)) >= sum(
                map(len, current_summaries)
            ):
                raise RuntimeError(
                    "Unable to reduce multiple document summaries."
                )

            current_summaries = next_summaries

        return await generate_limited(
            prompt=MULTI_DOC_REDUCE_PROMPT.format(
                summaries=combined
            ),
            instructions=(
                "Summarize multiple documents accurately, "
                "preserving their individual identities."
            ),
        )


    async def summarize_document(
        self,
        source: str,
        summaries: list[str],
        semaphore: asyncio.Semaphore | None = None,
    ) -> dict[str, str]:
        """Generate a final summary for one document."""

        summary = await self.reduce_summary(
            summaries,
            semaphore=semaphore,
        )

        return {
            "filename": source,
            "summary": summary,
        }

    async def summarize_documents(self) -> dict[str, Any]:
        """Summarize all uploaded documents."""

        chunks = vector_store.get_all_chunks()

        if not chunks:
            return {
                "message": SUMMARY_NO_DOCUMENTS_MESSAGE,
                "summaries": [],
            }

        grouped_documents = self.group_chunks(chunks)

        # Map stage.
        mapped_summaries = await self.map_documents(
            grouped_documents
        )

        reduction_semaphore = asyncio.Semaphore(
            max(1, settings.MAP_CONCURRENCY)
        )

        # Reduce stage: one final summary per PDF.
        document_summaries = await asyncio.gather(
            *[
                self.summarize_document(
                    source=source,
                    summaries=summaries,
                    semaphore=reduction_semaphore,
                )
                for source, summaries in mapped_summaries.items()
            ]
        )
        # Multi-document reduction.
                # Multi-document reduction.
        if len(document_summaries) > 1:

            overall_summary = await self.reduce_multiple_documents(
                document_summaries=document_summaries,
                semaphore=reduction_semaphore,
            )

        else:
            overall_summary = document_summaries[0]["summary"]
        logger.info(
            "Summarization completed for %d documents.",
            len(document_summaries),
        )

        return {
            "message": "Summarization completed successfully.",
            "document_count": len(document_summaries),
            "summaries": document_summaries,
            "overall_summary": overall_summary,
        }


summarization_agent = SummarizationAgent()
