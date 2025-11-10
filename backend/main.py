from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from .knowledge_agent import KnowledgeAgent
from .router_agent import RouterAgent
from .support_agent import SupportAgent
from .custom_agent import CustomAgent
from dotenv import load_dotenv
import httpx
import os
from .gemini_agent import GeminiAgent
from .models import LLMDecision
from .mcpo_tools import MCP_TOOLS, TOOLS

load_dotenv()

app = FastAPI(title="Agent API")

INFINITEPAY_PAGES = [
    "https://www.infinitepay.io",
    "https://www.infinitepay.io/maquininha",
    "https://www.infinitepay.io/maquininha-celular",
    "https://www.infinitepay.io/tap-to-pay",
    "https://www.infinitepay.io/pdv",
    "https://www.infinitepay.io/receba-na-hora",
    "https://www.infinitepay.io/gestao-de-cobranca-2",
    "https://www.infinitepay.io/gestao-de-cobranca",
    "https://www.infinitepay.io/link-de-pagamento",
    "https://www.infinitepay.io/loja-online",
    "https://www.infinitepay.io/boleto",
    "https://www.infinitepay.io/conta-digital",
    "https://www.infinitepay.io/pix",
    "https://www.infinitepay.io/emprestimo",
    "https://www.infinitepay.io/cartao",
    "https://www.infinitepay.io/rendimento",
]

knowledge_agent = KnowledgeAgent()
support_agent = SupportAgent()
custom_agent = CustomAgent()
router_agent = RouterAgent(knowledge_agent, support_agent, custom_agent)

class SwarmRequest(BaseModel):
    message: str
    user_id: str

class SwarmResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    used_retrieval: bool
    user_id: str
    tools_used: Optional[List[str]] = None

class MCPInvokeRequest(BaseModel):
    tool_id: str
    parameters: dict

@app.post("/swarm", response_model=SwarmResponse)
async def swarm_chat(req: SwarmRequest):
    try:
        result = await router_agent.route_and_respond(req.message, user_id=req.user_id)
        return SwarmResponse(
            answer=result["answer"],
            sources=result["sources"],
            used_retrieval=result["used_retrieval"],
            user_id=req.user_id,
            tools_used=result.get("tools_used") if isinstance(result, dict) else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/invoke")
async def invoke_tool(request: MCPInvokeRequest):
    tool_id = request.tool_id
    params = request.parameters or {}

    if tool_id not in TOOLS:
        return {"error": f"Ferramenta '{tool_id}' não reconhecida."}

    func = TOOLS[tool_id]
    return await func(params)

