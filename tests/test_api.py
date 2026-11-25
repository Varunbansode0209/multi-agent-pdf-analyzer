
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import backend.routes.upload as upload_module
import backend.routes.documents as documents_module
import backend.routes.chat as chat_module

from backend.main import app
from backend.ingestion.pdf_loader import InvalidPDFError
from backend.core.config import settings


@pytest.fixture
def client():
    return TestClient(app)


# HEALTH ENDPOINT

def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "multi-agent-pdf-analyzer",
    }


# PDF UPLOAD ENDPOINT

def test_upload_valid_pdf(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)

    def fake_ingest(path, filename):
        assert path.endswith(".pdf")
        assert filename == "sample.pdf"
        assert open(path, "rb").read() == b"%PDF-1.4 test content"

        return {
            "status": "success",
            "filename": filename,
            "pages": 2,
            "chunks": 5,
        }

    monkeypatch.setattr(
        upload_module.document_service,
        "ingest_document",
        fake_ingest,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"%PDF-1.4 test content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["pages"] == 2

    # Temporary upload file must be removed.
    assert list(tmp_path.iterdir()) == []


def test_upload_rejects_non_pdf_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"Some text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."


def test_upload_rejects_empty_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "empty.pdf",
                b"",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_upload_rejects_oversized_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "large.pdf",
                b"%PDF-1.4 content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "PDF exceeds the upload size limit."


def test_upload_handles_invalid_pdf(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)

    def fake_ingest(path, filename):
        raise InvalidPDFError("Invalid PDF structure.")

    monkeypatch.setattr(
        upload_module.document_service,
        "ingest_document",
        fake_ingest,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "invalid.pdf",
                b"Not a valid PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid PDF structure."
    assert list(tmp_path.iterdir()) == []


def test_upload_handles_value_error(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)

    def fake_ingest(path, filename):
        raise ValueError("Invalid document data.")

    monkeypatch.setattr(
        upload_module.document_service,
        "ingest_document",
        fake_ingest,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"%PDF-1.4 content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid document data."


def test_upload_handles_unexpected_error(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)

    def fake_ingest(path, filename):
        raise RuntimeError("Unexpected database error.")

    monkeypatch.setattr(
        upload_module.document_service,
        "ingest_document",
        fake_ingest,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"%PDF-1.4 content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Document ingestion failed."


# DOCUMENT LIST ENDPOINT

def test_list_documents(client, monkeypatch):
    monkeypatch.setattr(
        documents_module.document_service,
        "list_documents",
        Mock(return_value=[
            {
                "source": "sample.pdf",
                "pages": 3,
                "chunks": 8,
            }
        ]),
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == [
        {
            "doc_id": "sample.pdf",
            "filename": "sample.pdf",
            "pages": 3,
            "chunks": 8,
        }
    ]


def test_list_documents_when_empty(client, monkeypatch):
    monkeypatch.setattr(
        documents_module.document_service,
        "list_documents",
        Mock(return_value=[]),
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


# DOCUMENT DELETE ENDPOINT

def test_delete_document_success(client, monkeypatch):
    monkeypatch.setattr(
        documents_module.document_service,
        "delete_document",
        Mock(return_value=True),
    )

    response = client.delete("/documents/sample.pdf")

    assert response.status_code == 200
    assert response.json() == {
        "status": "deleted",
        "filename": "sample.pdf",
    }


def test_delete_document_not_found(client, monkeypatch):
    monkeypatch.setattr(
        documents_module.document_service,
        "delete_document",
        Mock(return_value=False),
    )

    response = client.delete("/documents/unknown.pdf")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


# CHAT ENDPOINT

def test_chat_endpoint_success(client, monkeypatch):
    async def fake_process_message(message, session_id):
        return {
            "session_id": session_id,
            "answer": "The document discusses machine learning.",
            "route": "answer_question",
            "sources": [],
            "retrieved_chunks": 2,
        }

    monkeypatch.setattr(
        chat_module.chat_service,
        "process_message",
        fake_process_message,
    )

    response = client.post(
        "/chat/",
        json={
            "message": "What is machine learning?",
            "session_id": "session-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-1"
    assert response.json()["route"] == "answer_question"
    assert response.json()["retrieved_chunks"] == 2


def test_chat_endpoint_rejects_invalid_request(client):
    response = client.post(
        "/chat/",
        json={
            "message": "",
            "session_id": "session-1",
        },
    )

    assert response.status_code == 422


def test_chat_endpoint_handles_value_error(client, monkeypatch):
    async def fake_process_message(message, session_id):
        raise ValueError("Message cannot be empty.")

    monkeypatch.setattr(
        chat_module.chat_service,
        "process_message",
        fake_process_message,
    )

    response = client.post(
        "/chat/",
        json={
            "message": "Hello",
            "session_id": "session-1",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Message cannot be empty."


def test_chat_endpoint_handles_unexpected_error(client, monkeypatch):
    async def fake_process_message(message, session_id):
        raise RuntimeError("LLM API unavailable.")

    monkeypatch.setattr(
        chat_module.chat_service,
        "process_message",
        fake_process_message,
    )

    response = client.post(
        "/chat/",
        json={
            "message": "Hello",
            "session_id": "session-1",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An error occurred while processing your message."
    )


# CHAT DELETE ENDPOINT

def test_clear_chat_success(client, monkeypatch):
    monkeypatch.setattr(
        chat_module.chat_service,
        "clear_conversation",
        Mock(return_value=True),
    )

    response = client.delete("/chat/session-1")

    assert response.status_code == 200
    assert response.json()["message"] == "Conversation cleared."


def test_clear_chat_missing_session(client, monkeypatch):
    monkeypatch.setattr(
        chat_module.chat_service,
        "clear_conversation",
        Mock(return_value=False),
    )

    response = client.delete("/chat/unknown-session")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "No conversation found for this session."
    )
