
from typing import Any

import chromadb

from backend.core.config import settings


class ChromaVectorStore:

    def __init__(self):

        settings.validate_embedding_backend()

        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.ACTIVE_COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def to_similarity(distance: float) -> float:
        """Convert cosine distance into cosine similarity."""

        return max(0.0, min(1.0, 1.0 - distance))

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:

        if not documents:
            return

        if not (
            len(documents)
            == len(embeddings)
            == len(metadatas)
            == len(ids)
        ):
            raise ValueError(
                "Documents, embeddings, metadata and IDs "
                "must have equal lengths."
            )

        dimensions = {len(vector) for vector in embeddings}

        if len(dimensions) != 1:
            raise ValueError("All embeddings must have the same dimensions.")

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = settings.TOP_K,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        total = self.collection.count()

        if total == 0:
            return []

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, total),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where is not None:
            query_args["where"] = where

        results = self.collection.query(**query_args)

        matches = []

        for i, document in enumerate(results["documents"][0]):

            distance = results["distances"][0][i]

            matches.append({
                "id": results["ids"][0][i],
                "text": document,
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity": self.to_similarity(distance),
            })

        return matches

    def get_all_chunks(
        self,
        source: str | None = None
    ) -> list[dict[str, Any]]:

        kwargs = {
            "include": ["documents", "metadatas"]
        }

        if source:
            kwargs["where"] = {"source": source}

        result = self.collection.get(**kwargs)

        chunks = [
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata,
            }
            for chunk_id, document, metadata in zip(
                result["ids"],
                result["documents"],
                result["metadatas"],
            )
        ]

        chunks.sort(
            key=lambda item: (
                item["metadata"]["source"],
                item["metadata"]["page"],
                item["metadata"]["chunk_index"],
            )
        )

        return chunks

    def list_sources(self) -> list[dict[str, Any]]:

        result = self.collection.get(
            include=["metadatas"]
        )

        stats = {}

        for metadata in result["metadatas"]:

            source = metadata["source"]

            if source not in stats:
                stats[source] = {
                    "source": source,
                    "chunks": 0,
                    "pages": set(),
                }

            stats[source]["chunks"] += 1
            stats[source]["pages"].add(metadata["page"])

        return [
            {
                "source": item["source"],
                "chunks": item["chunks"],
                "pages": len(item["pages"]),
            }
            for item in stats.values()
        ]

    def delete_by_source(self, source: str) -> None:

        self.collection.delete(
            where={"source": source}
        )

    def count(self) -> int:
        return self.collection.count()


vector_store = ChromaVectorStore()
