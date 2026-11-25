
import hashlib
from pathlib import Path

from backend.core.config import settings
from backend.ingestion.pdf_loader import extract_pdf_pages
from backend.ingestion.text_splitter import split_pages_into_chunks
from backend.ingestion.embeddings import embedding_generator
from backend.vectorstore.chroma_store import vector_store


def sanitize_filename(filename: str) -> str:
    """Remove directory components from an uploaded filename."""
    return Path(filename.replace("\\", "/")).name


class DocumentService:

    def __init__(self):
        settings.ensure_dirs()

    def ingest_document(
        self,
        file_path: str,
        original_filename: str | None = None,
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        source_name = (
            sanitize_filename(original_filename)
            if original_filename
            else path.name
        )

        if not source_name or not source_name.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        # 1. Extract pages.
        pages = extract_pdf_pages(str(path))

        # Preserve the real uploaded filename in page metadata.
        for page in pages:
            page["metadata"]["source"] = source_name

        # 2. Split into chunks.
        chunks = split_pages_into_chunks(pages)

        if not chunks:
            raise ValueError("No text chunks generated from PDF.")

        # 3. Generate embeddings.
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_generator.generate_embeddings(texts)

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match chunk count.")

        # 4. Generate a stable document ID.
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        document_id = hashlib.sha256(
            f"{source_name}:{file_hash}".encode()
        ).hexdigest()[:16]

        # 5. Prepare chunk IDs and metadata.
        ids = [
            f"{document_id}_p{chunk['metadata']['page']}_c{chunk['metadata']['chunk_index']}"
            for chunk in chunks
        ]

        metadatas = [
            {
                **chunk["metadata"],
                "document_id": document_id,
            }
            for chunk in chunks
        ]

        # 6. Insert the new version first.
        vector_store.add_documents(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        # 7. Delete old versions only after successful insertion.
        existing_chunks = vector_store.collection.get(
            where={"source": source_name},
            include=["metadatas"],
        )

        old_ids = [
            chunk_id
            for chunk_id in existing_chunks["ids"]
            if chunk_id not in ids
        ]

        if old_ids:
            vector_store.collection.delete(ids=old_ids)

        return {
            "document_id": document_id,
            "filename": source_name,
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "success",
        }

    def list_documents(self) -> list[dict]:
        return vector_store.list_sources()

    def delete_document(self, filename: str) -> bool:
        source_name = sanitize_filename(filename)

        existing_sources = {
            document["source"]
            for document in vector_store.list_sources()
        }

        if source_name not in existing_sources:
            return False

        vector_store.delete_by_source(source_name)
        return True


document_service = DocumentService()
