
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from backend.services.document_service import document_service

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.get("")
async def list_documents():
    documents = await run_in_threadpool(
        document_service.list_documents
    )

    return [
        {
            "doc_id": document["source"],
            "filename": document["source"],
            "pages": document["pages"],
            "chunks": document["chunks"],
        }
        for document in documents
    ]


@router.delete("/{filename}")
async def delete_document(filename: str):
    deleted = await run_in_threadpool(
        document_service.delete_document,
        filename,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "status": "deleted",
        "filename": filename,
    }
