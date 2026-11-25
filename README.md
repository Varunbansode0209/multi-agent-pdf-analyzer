# Multi-Agent PDF Analyzer

An AI-powered PDF analysis application that uses autonomous agent orchestration, Retrieval-Augmented Generation (RAG), and document summarization to help users interact with multiple PDF documents through natural language.

## Overview

Multi-Agent PDF Analyzer allows users to upload PDF documents, ask questions about their contents, and generate document summaries through a conversational interface.

The application uses the OpenAI Agents SDK to route requests to specialized agents:

1. **RAG Agent** – Retrieves relevant document passages and generates context-grounded answers.
2. **Summarization Agent** – Generates structured summaries of long documents using a map-reduce approach.

An orchestrator determines which capability is appropriate for the user's request, so users do not need to manually select an operating mode.

## Features

- Upload and manage multiple PDF documents.
- Extract and process PDF text.
- Split documents into overlapping text chunks.
- Generate local embeddings using Sentence Transformers.
- Store and retrieve document embeddings using ChromaDB.
- Answer questions using retrieved document context.
- Display source references with answers.
- Summarize individual documents and multiple documents.
- Use map-reduce summarization for long documents.
- Automatically route requests through the agent orchestrator.
- Maintain conversational history using session IDs.
- Delete uploaded documents.
- Clear conversations.
- Display processing status and errors through the frontend.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Agent orchestration | OpenAI Agents SDK |
| LLM | OpenAI API |
| PDF processing | PyMuPDF |
| Text splitting | Custom text splitter |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| HTTP client | Requests |
| Testing | Pytest |

## Architecture

```text
                  User
                   |
                   v
             Streamlit UI
                   |
                   v
              FastAPI API
                   |
                   v
          Agent Orchestrator
                   |
             Intent Routing
              /         \
             v           v
         RAG Agent   Summarization
             |           Agent
             v             |
       Query Embedding     v
             |         Map-Reduce
             v         Processing
         ChromaDB           |
             |              v
             v          LLM API
       Relevant Chunks       |
             |              |
             v              v
          LLM API       Final Summary
             |
             v
       Grounded Answer
```

### Document Ingestion

1. The user uploads a PDF through Streamlit.
2. FastAPI receives the document.
3. PyMuPDF extracts its text.
4. The text splitter divides it into manageable chunks with overlap.
5. Sentence Transformer generates embeddings locally.
6. ChromaDB stores the embeddings and associated document metadata.

### Question Answering

1. The user submits a natural-language question.
2. The orchestrator routes the request to the RAG Agent.
3. The query is embedded using the configured embedding model.
4. ChromaDB retrieves relevant document chunks.
5. Retrieved context is passed to the LLM.
6. The agent generates an answer grounded in the retrieved information and returns source references.

### Document Summarization

1. The orchestrator routes summarization requests to the Summarization Agent.
2. The document is processed in manageable batches.
3. The map stage generates intermediate summaries.
4. The reduce stage combines and refines those summaries.
5. The application returns individual document summaries and an overall summary when applicable.

## Project Structure

```text
multi-agent-pdf-analyzer/
├── frontend/
│   └── app.py
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── rag_agent.py
│   │   └── summarization_agent.py
│   ├── core/
│   │   ├── config.py
│   │   └── prompts.py
│   ├── ingestion/
│   │   ├── embeddings.py
│   │   ├── pdf_loader.py
│   │   └── text_splitter.py
│   ├── routes/
│   │   ├── chat.py
│   │   ├── documents.py
│   │   └── upload.py
│   ├── services/
│   │   ├── chat_service.py
│   │   └── document_service.py
│   ├── vectorstore/
│   │   └── chroma_store.py
│   └── main.py
├── data/
│   ├── uploads/
│   └── vectorstore/
├── docs/
├── logs/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.11
- An OpenAI API key
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Varunbansode0209/multi-agent-pdf-analyzer.git
cd multi-agent-pdf-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example configuration:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

Keep your API key private. Do not commit the `.env` file.

The application uses local Sentence Transformer embeddings by default. The embedding model may be downloaded automatically on first use.

## Running the Application

The frontend and backend run separately.

### Terminal 1: Start FastAPI

```bash
uvicorn backend.main:app --reload
```

Backend:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

Health endpoint:

`http://127.0.0.1:8000/health`

### Terminal 2: Start Streamlit

```bash
streamlit run frontend/app.py
```

Frontend:

`http://localhost:8501`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents/upload` | Upload and process a PDF |
| GET | `/documents` | List uploaded documents |
| DELETE | `/documents/{filename}` | Delete a document |
| POST | `/chat/` | Submit a question or summarization request |
| DELETE | `/chat/{session_id}` | Clear a conversation |
| GET | `/health` | Check backend health |

## Testing

Run the complete test suite:

```bash
python -m pytest -v
```

**Latest verified test result: 117 passed.**

The test suite covers agent behavior, orchestration, API routes, ingestion, chat services, and retrieval functionality.

Manual integration testing also verified:

- Backend connectivity.
- PDF upload and indexing.
- Question answering with retrieved context.
- Individual and overall document summarization.
- Document deletion.
- Conversation clearing.

## Configuration

Application settings can be adjusted through environment variables in `.env`.

Important configuration parameters include:

| Variable | Purpose |
|---|---|
| `LLM_MODEL` | LLM model used by the agents |
| `EMBEDDING_BACKEND` | Embedding provider |
| `EMBEDDING_MODEL` | Sentence Transformer model |
| `CHUNK_SIZE` | Maximum text chunk size |
| `CHUNK_OVERLAP` | Overlap between chunks |
| `TOP_K` | Number of retrieved chunks |
| `MAP_BATCH_CHARS` | Character limit for summarization batches |
| `MAP_CONCURRENCY` | Maximum concurrent map operations |
| `REDUCE_MAX_CHARS` | Character limit for reduction processing |
| `MAX_UPLOAD_MB` | Maximum permitted PDF upload size |

Refer to `.env.example` for the complete configuration.

## Current Limitations

- Conversation history is maintained in backend memory and is not persistent across backend restarts.
- The vector database is stored locally.
- An active OpenAI API key and API access are required for LLM operations.
- Authentication and production deployment are outside the current implementation.

## Author

**Varun Bansode**

GitHub: https://github.com/Varunbansode0209

LinkedIn: https://www.linkedin.com/in/varun-bansode-75176a355/

## Assignment

AI Engineer Technical Assignment – Multi-Agent PDF Analyzer
