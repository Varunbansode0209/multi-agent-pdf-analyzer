
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.routes import upload, documents,chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()
    settings.ensure_dirs()
    yield


app = FastAPI(
    title="Multi-Agent PDF Analyzer",
    description="RAG and document summarization using specialized AI agents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "multi-agent-pdf-analyzer",
    }
