import os
import httpx

MCP_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

class MCPClient:
    """Cliente para interagir com o servidor MCP (ex: FastAPI + Gemini)."""

    def __init__(self, base_url: str = MCP_BASE_URL):
        self.base_url = base_url.rstrip("/")

    async def invoke_tool(self, tool_id: str, parameters: dict):
        """Invoca uma ferramenta registrada no servidor MCP."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/mcp/invoke",
                    json={"tool_id": tool_id, "parameters": parameters},
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                return {"error": f"Erro de rede: {e}"}
            except httpx.HTTPStatusError as e:
                return {"error": f"Erro HTTP: {e.response.status_code} - {e.response.text}"}
