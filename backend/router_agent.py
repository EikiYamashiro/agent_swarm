from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel
import json

from .knowledge_agent import KnowledgeAgent
from .support_agent import SupportAgent
from .mcpo_tools import GeminiAgent, get_knowledge_tool
from .custom_agent import CustomAgent
from .mcp_client import MCPClient
from .models import LLMDecision


class RouterAgent:
    """Agent responsible for routing messages to specialized sub-agents (Support, Knowledge, Custom, etc)."""

    FORMATTING_PROMPT = """You are a helpful assistant that formats responses using provided context.
Format the following response to be clear, professional, and well-structured.

Context: {context}
Initial Answer: {initial_answer}

Please:
1. Maintain factual accuracy
2. Improve clarity and readability
3. Use paragraphs where appropriate
4. Keep a professional, user-friendly tone
"""

    def __init__(
        self,
        knowledge_agent: KnowledgeAgent,
        support_agent: SupportAgent,
        custom_agent: Optional[CustomAgent] = None,
    ):
        self.knowledge_agent = knowledge_agent
        self.support_agent = support_agent
        self.custom_agent = custom_agent
        self.gemini_agent = GeminiAgent()

    async def route_and_respond(self, message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        observations, last_output = [], None
        max_steps = 3

        try:
            knowledge = get_knowledge_tool()
            parts = []
            for i, (url, summary) in enumerate(knowledge.items()):
                s = (summary or "").replace("\n", " ")
                parts.append(f"- {url}: {s[:200]}{'...' if len(s) > 200 else ''}")
            knowledge_context = "\n".join(parts) if parts else "(no knowledge entries)"
        except Exception:
            knowledge_context = "(unable to load knowledge)"

        mcp_client = MCPClient()

        for step in range(max_steps):
            prompt_data = {
                "message": message,
                "knowledge_context": knowledge_context
            }

            route_response = await mcp_client.invoke_tool("route_message", prompt_data)

            route_result = route_response.get("result", {})

            if isinstance(route_result, dict) and "selected_agent" in route_result:
                decision = LLMDecision(**route_result)
            else:
                decision = LLMDecision(
                    selected_agent="DIRECT",
                    is_final=True,
                    reasoning=f"Fallback routing (invalid response): {route_result}"
                )

            sel = decision.selected_agent.strip().upper()
            result = await self._dispatch(sel, message, user_id)
            last_output = result

            obs_text = result.get("answer") if isinstance(result, dict) else str(result)
            observations.append(f"Step {step + 1} via {sel}: {obs_text}")

            if decision.is_final:
                return await self._format_final(message, observations, last_output, used_retrieval=(sel == "RETRIEVE"))

        return await self._format_final(message, observations, last_output, used_retrieval=False)

    async def _dispatch(self, sel: str, message: str, user_id: Optional[str]) -> Dict[str, Any]:
        if sel == "SUPPORT" and self.support_agent:
            return await self.support_agent.handle_inquiry(user_id or "unknown", message)

        if sel == "ADD_KNOWLEDGE" and self.custom_agent:
            try:
                self.knowledge_agent._indexed = False
            except Exception:
                pass
            return await self.custom_agent.handle_add_request(user_id or "unknown", message)

        if sel == "RETRIEVE":
            return await self.knowledge_agent.answer(message)

        direct = self.gemini_agent.generate(message)
        return {"answer": direct.get("answer") or direct.get("text"), "tools_used": []}

    async def _format_final(self, message: str, observations: list, last_output: Dict[str, Any], used_retrieval: bool):
        context = "\n\n".join(observations)
        prompt = self.FORMATTING_PROMPT.format(context=context, initial_answer=message)
        final = self.gemini_agent.generate(prompt)

        final_text = final.get("answer") or final.get("text") if isinstance(final, dict) else str(final)
        sources = last_output.get("sources", []) if isinstance(last_output, dict) else []

        return {
            "answer": final_text,
            "sources": sources,
            "used_retrieval": used_retrieval,
        }
