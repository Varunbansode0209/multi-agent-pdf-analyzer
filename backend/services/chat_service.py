
import logging
from typing import Any

from backend.agents.orchestrator import orchestrator
from backend.core.config import settings


logger = logging.getLogger(__name__)


class ChatService:

    def __init__(self):
        # In-memory conversation storage.
        # Each session has its own conversation history.
        self.conversations: dict[str, list[dict[str, str]]] = {}

    async def process_message(
        self,
        message: str,
        session_id: str,
    ) -> dict[str, Any]:

        if not message or not message.strip():
            raise ValueError("Message cannot be empty.")

        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty.")

        history = self.conversations.get(session_id, [])

        try:
            result = await orchestrator.process_message(
                message=message,
                history=history,
            )

            # Store the conversation after successful processing.
            history.append({
                "role": "user",
                "content": message.strip(),
            })

            history.append({
                "role": "assistant",
                "content": result.get(
                    "answer",
                    result.get("overall_summary", ""),
                ),
            })

            # Retain only recent conversation turns.
            max_messages = settings.MAX_HISTORY_TURNS * 2

            self.conversations[session_id] = history[-max_messages:]

            return {
                "session_id": session_id,
                **result,
            }

        except Exception:
            logger.exception(
                "Chat processing failed for session %s",
                session_id,
            )
            raise

    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session."""

        return self.conversations.pop(session_id, None) is not None


chat_service = ChatService()
