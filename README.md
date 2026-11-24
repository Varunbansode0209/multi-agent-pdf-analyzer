
# Multi-Agent Reasoning System for PDF Analysis

An intelligent PDF analysis application powered by
Retrieval-Augmented Generation (RAG), multi-agent
orchestration, and Large Language Models.

## Overview

The application allows users to upload multiple PDF
documents and interact with them through natural language.

It uses an autonomous agent orchestration system to
dynamically delegate requests to two specialized agents:

1. RAG Agent - Answers questions using relevant
   information retrieved from uploaded PDFs.

2. Summarization Agent - Generates coherent,
   context-aware summaries of uploaded documents.

## Architecture

- Frontend: Streamlit
- Backend: FastAPI
- Agent Framework: OpenAI Agents SDK
- PDF Processing: PyMuPDF
- Embeddings: Sentence Transformers
- Vector Database: ChromaDB
- LLM: OpenAI API

## Key Features

- Multiple PDF document uploads
- Semantic document retrieval
- Context-grounded question answering
- Abstractive document summarization
- Autonomous agent selection
- Conversational interface
- Source and page references

## Project Structure

```text
frontend/
backend/
    agents/
    ingestion/
    routes/
    services/
    vectorstore/
    core/
tests/
docs/
data/
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd codesis-multi-agent-pdf-analyzer
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables using `.env.example`.

## Running the Application

Instructions will be added after implementation.

## Testing

Instructions and test results will be added after implementation.

## Deployment

Deployment information will be added after implementation.

## Author

Varun

## Assignment

AI Engineer Technical Assignment
