import re
from typing import Dict, Any, List


class CustomAgent:
    """Handles custom commands such as adding a URL to the knowledge base.

    The agent uses the MCP gemini wrapper to interpret the user's intent and
    calls the `mcp_tools.add_knowledge_url` tool to scrape and persist the page.
    """
    def __init__(self):
        try:
            from .mcpo_tools import GeminiAgent
            self.gemini = GeminiAgent()
        except Exception:
            self.gemini = None

    def _extract_first_url(self, text: str) -> str:
        m = re.search(r"(https?://\S+)", text)
        if not m:
            return ""
        url = m.group(1).rstrip('.,)')
        return url

    async def handle_add_request(self, user_id: str, message: str) -> Dict[str, Any]:
        """If message contains a URL and the user intends to add it to knowledge,
        call the add_knowledge_url tool and return a friendly message.
        """
        tools_used: List[str] = []

        url = self._extract_first_url(message)
        if not url:
            return {"answer": "Não encontrei uma URL na mensagem.", "tools_used": tools_used}

        # Ask LLM whether this is a request to add the URL to the knowledge base
        confirm = None
        if self.gemini:
            prompt = (
                f"O usuário pediu: \"{message}\".\n\n" 
                f"A mensagem contém a solicitação para ADICIONAR a URL {url} ao repositório de conhecimento? "
                "Responda apenas com SIM ou NÃO."
            )
            try:
                resp = await self.gemini.tool_gemini_generate({"prompt": prompt, "max_tokens": 100})
                if isinstance(resp, dict) and "result" in resp:
                    result = resp["result"]
                    if isinstance(result, dict) and "candidates" in result:
                        confirm = result["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
                    else:
                        confirm = None
                else:
                    confirm = None
            except Exception:
                confirm = None

        # fallback heuristic if LLM not available or ambiguous
        if not confirm:
            lower = message.lower()
            if any(w in lower for w in ("adicione", "adicionar", "adiciona", "add", "incluir")):
                confirm = "SIM"
            else:
                confirm = "NAO"

        if confirm.startswith("S"):
            # perform the add via mcp_tools
            try:
                from .mcpo_tools import add_knowledge_url_tool
                entry = add_knowledge_url_tool({"url": url})
                tools_used.append("add_knowledge_url")
                return {"answer": f"URL adicionada ao knowledge: {url}", "tools_used": tools_used, "entry": entry}
            except Exception as e:
                return {"answer": f"Falha ao adicionar a URL: {str(e)}", "tools_used": tools_used}

        return {"answer": "Não vou adicionar o link ao knowledge.", "tools_used": tools_used}
