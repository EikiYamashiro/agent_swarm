from typing import Dict, Any, List, Optional
from .mcpo_tools import (
    get_user_profile_tool,
    create_support_ticket_tool,
    GeminiAgent,
)


class SupportAgent:
    """Agent that handles customer support inquiries using only MCP tools for data and LLM calls."""

    def __init__(self):
        # Initialize Gemini agent from mcpo_tools
        self.gemini = GeminiAgent()

    async def handle_inquiry(self, user_id: str, message: str) -> Dict[str, Any]:
        """Process a support request for a user.

        Returns a dict with:
        - answer: str
        - tools_used: List[str]
        - ticket: Optional[Dict]
        """
        tools_used: List[str] = []
        ticket = None

        # Get user profile
        profile = get_user_profile_tool({"user_id": user_id})
        if profile and "error" not in profile:
            tools_used.append("get_user_profile")

        # Build context from available information
        context_parts: List[str] = []
        if profile and isinstance(profile, dict):
            context_parts.append("User Profile:\n" + "\n".join([f"{k}: {v}" for k, v in profile.items() if k != 'transactions']))

        context = "\n\n".join(context_parts) if context_parts else "No user-specific context available."

        # Generate response using Gemini
        prompt = (
            "You are a customer support assistant for Infinitepay. Use the user context below to answer the user's question. "
            "If the user appears to report an issue that requires escalation (e.g., disputed charge, missing funds, account suspension), recommend creating a support ticket. "
            "Keep privacy in mind and avoid revealing sensitive tokens or PII beyond what's necessary.\n\n"
            f"User question: {message}\n\nContext:\n{context}\n\nResponse:" 
        )
        
        gen_response = await self.gemini.tool_gemini_generate({"prompt": prompt, "max_tokens": 300})
        answer_text = ""
        if isinstance(gen_response, dict) and "result" in gen_response:
            result = gen_response["result"]
            if isinstance(result, dict) and "candidates" in result:
                answer_text = result["candidates"][0]["content"]["parts"][0]["text"]

        if not answer_text:
            answer_text = "I apologize, but I was unable to generate a proper response. Please try again."

        # Create ticket if needed
        lower = message.lower()
        if any(k in lower for k in ("dispute", "chargeback", "refund", "unauthorized", "stolen", "missing")):
            ticket_subject = f"Support request: {message[:60]}"
            ticket_body = f"Auto-created from support agent. User: {user_id}\nQuestion: {message}\nContext:\n{context}"
            ticket = create_support_ticket_tool({
                "user_id": user_id,
                "subject": ticket_subject,
                "body": ticket_body
            })
            if ticket and "error" not in ticket:
                tools_used.append("create_support_ticket")
                ticket_id = ticket.get("ticket_id", "unknown")
                answer_text += f"\n\nA support ticket has been created with ID {ticket_id}. Our team will follow up."

        return {
            "answer": answer_text,
            "tools_used": tools_used,
            "ticket": ticket
        }
