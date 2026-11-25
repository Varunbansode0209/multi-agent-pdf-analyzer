
import os
import uuid
from urllib.parse import quote

import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 120


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent PDF Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []


# --------------------------------------------------
# API helpers
# --------------------------------------------------

def api_request(method, endpoint, **kwargs):
    """Send an HTTP request to the FastAPI backend."""

    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{endpoint}",
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )

        response.raise_for_status()

        if response.status_code == 204:
            return None

        return response.json()

    except requests.exceptions.Timeout:
        st.error("The backend request timed out.")
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the backend. "
            "Make sure FastAPI is running."
        )
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get(
                "detail",
                str(exc),
            )
        except ValueError:
            detail = str(exc)

        st.error(f"API error: {detail}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")

    return None


def refresh_documents():
    documents = api_request("GET", "/documents")

    if documents is not None:
        st.session_state.documents = documents


def upload_document(uploaded_file):
    """Upload one PDF to the backend."""

    return api_request(
        "POST",
        "/documents/upload",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        },
        timeout=300,
    )


# --------------------------------------------------
# Response rendering
# --------------------------------------------------

def render_sources(sources):
    if not sources:
        return

    with st.expander("Sources and references"):
        for index, source in enumerate(sources, start=1):
            if isinstance(source, dict):
                st.markdown(f"**Source {index}**")
                st.json(source)
            else:
                st.write(source)


def render_response(result):
    """Display RAG, conversation, and summary responses."""

    if result.get("answer"):
        st.markdown(result["answer"])

    if result.get("message"):
        st.info(result["message"])

    summaries = result.get("summaries", [])

    if summaries:
        st.markdown("### Document summaries")

        for item in summaries:
            filename = item.get("filename", "Document")
            summary = item.get("summary", "")

            with st.expander(filename):
                st.markdown(summary)

    overall_summary = result.get("overall_summary")

    if overall_summary:
        st.markdown("### Overall summary")
        st.markdown(overall_summary)

    if result.get("route") == "answer_question":
        st.caption(
            f"Retrieved chunks: "
            f"{result.get('retrieved_chunks', 0)}"
        )

    render_sources(result.get("sources", []))


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Multi-Agent PDF Analyzer")

st.caption(
    "Upload PDF documents, ask questions, "
    "and generate AI-powered summaries."
)

health = api_request("GET", "/health")

if health:
    st.caption("Backend connected")
else:
    st.warning(
        "Backend is unavailable. Start FastAPI before using the application."
    )


# --------------------------------------------------
# Sidebar: document management
# --------------------------------------------------

with st.sidebar:

    st.header("Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files.",
    )

    if st.button(
        "Upload selected PDFs",
        use_container_width=True,
        disabled=not uploaded_files,
    ):

        for uploaded_file in uploaded_files:

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):
                result = upload_document(uploaded_file)

            if result is not None:
                st.success(
                    f"Uploaded: {uploaded_file.name}"
                )

        refresh_documents()

    st.divider()

    st.subheader("Uploaded Documents")

    if st.button(
        "Refresh documents",
        use_container_width=True,
    ):
        refresh_documents()

    if not st.session_state.documents:
        refresh_documents()

    if st.session_state.documents:

        for document in st.session_state.documents:

            filename = document.get(
                "filename",
                "Unknown document",
            )

            with st.container(border=True):

                st.markdown(f"**{filename}**")

                st.caption(
                    f"Pages: {document.get('pages', 0)} | "
                    f"Chunks: {document.get('chunks', 0)}"
                )

                if st.button(
                    "Delete",
                    key=f"delete_{filename}",
                    use_container_width=True,
                ):

                    result = api_request(
                        "DELETE",
                        f"/documents/{quote(filename, safe='')}",
                    )

                    if result is not None:
                        st.success("Document deleted.")
                        refresh_documents()
                        st.rerun()

    else:
        st.info("No documents uploaded yet.")

    st.divider()

    st.subheader("Conversation")

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        api_request(
            "DELETE",
            f"/chat/{st.session_state.session_id}",
        )

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()


# --------------------------------------------------
# Main chat interface
# --------------------------------------------------

st.subheader("Chat with your documents")

# Display previous messages.

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_response(message["data"])


# Chat input.

user_input = st.chat_input(
    "Ask a question or request a document summary..."
)

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing your request..."):

            result = api_request(
                "POST",
                "/chat/",
                json={
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                },
            )

        if result is not None:

            render_response(result)

            st.session_state.messages.append({
                "role": "assistant",
                "data": result,
            })
