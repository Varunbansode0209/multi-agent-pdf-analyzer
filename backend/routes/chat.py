
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.chat_service import chat_service


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str | None = None
    message: str | None = None
    route: str
    sources: list[dict] = Field(default_factory=list)
    retrieved_chunks: int = 0
    document_count: int | None = None
    summaries: list[dict] = Field(default_factory=list)
    overall_summary: str | None = None


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a user message through the planner."""

    try:
        result = await chat_service.process_message(
            message=request.message,
            session_id=request.session_id,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception("Chat endpoint failed.")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your message.",
        )


@router.delete("/{session_id}")
async def clear_chat(session_id: str):
    """Clear conversation history for a session."""

    deleted = chat_service.clear_conversation(session_id)

    return {
        "message": (
            "Conversation cleared."
            if deleted
            else "No conversation found for this session."
        ),
        "session_id": session_id,
    }
