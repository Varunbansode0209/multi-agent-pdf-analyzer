
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    function_tool,
    set_default_openai_key,
)

from backend.core.config import settings
from backend.core.prompts import PLANNER_INSTRUCTIONS
from backend.agents.rag_agent import rag_agent
from backend.agents.summarization_agent import summarization_agent


logger = logging.getLogger(__name__)


@dataclass
class PlannerContext:
    """Per-request context shared with planner tools."""

    history: list[dict[str, str]] = field(default_factory=list)
    selected_tool: str | None = None
    tool_result: dict[str, Any] | None = None


@function_tool
async def answer_question(
    ctx: RunContextWrapper[PlannerContext],
    question: str,
) -> str:
    """Answer a specific question using uploaded PDF documents."""

    result = await asyncio.to_thread(
        rag_agent.answer_question,
        question,
        ctx.context.history,
    )

    ctx.context.selected_tool = "answer_question"
    ctx.context.tool_result = result

    return json.dumps(result, ensure_ascii=False)


@function_tool
async def summarize_documents(
    ctx: RunContextWrapper[PlannerContext],
) -> str:
    """Generate summaries of all uploaded PDF documents."""

    result = await summarization_agent.summarize_documents()

    ctx.context.selected_tool = "summarize_documents"
    ctx.context.tool_result = result

    return json.dumps(result, ensure_ascii=False)


class Orchestrator:

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        # Configure the Agents SDK with the project's API key.
        set_default_openai_key(settings.OPENAI_API_KEY)

        self.planner = Agent(
            name="PDF Planner",
            instructions=PLANNER_INSTRUCTIONS,
            model=settings.LLM_MODEL,
            tools=[
                answer_question,
                summarize_documents,
            ],
        )

    @staticmethod
    def format_history(
        history: list[dict[str, str]] | None,
    ) -> str:
        """Format recent conversation history for the planner."""

        if not history:
            return ""

        recent_history = history[-settings.MAX_HISTORY_TURNS * 2:]

        lines = []

        for item in recent_history:
            role = item.get("role", "").lower()
            content = item.get("content", "").strip()

            if role in {"user", "assistant"} and content:
                lines.append(f"{role.capitalize()}: {content}")

        if not lines:
            return ""

        return "Previous conversation:\n" + "\n".join(lines)

    async def process_message(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Route a user message to the appropriate PDF agent."""

        if not message or not message.strip():
            raise ValueError("Message cannot be empty.")

        context = PlannerContext(
            history=history or [],
        )

        previous_conversation = self.format_history(history)

        planner_input = (
            f"{previous_conversation}\n\n"
            f"Current user message: {message}"
            if previous_conversation
            else message
        )

        result = await Runner.run(
            self.planner,
            input=planner_input,
            context=context,
            max_turns=4,
        )

        # Return the actual tool result when a tool was called.
        if context.tool_result is not None:
            response = context.tool_result.copy()
            response["route"] = context.selected_tool
            response["planner_output"] = str(result.final_output)

            return response

        # Greetings and unrelated messages do not need a tool.
        return {
            "answer": str(result.final_output),
            "route": "conversation",
            "sources": [],
            "retrieved_chunks": 0,
        }


orchestrator = Orchestrator()
