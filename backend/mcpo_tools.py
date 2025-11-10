from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from pydantic import BaseModel
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import httpx
import asyncio
from typing import Dict, Any
from pydantic import BaseModel
from .knowledge_agent import KnowledgeAgent
from .gemini_agent import GeminiAgent
from .models import LLMDecision

# ---------------------------
# Diretórios e arquivos locais
# ---------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
MESSAGES_FILE = DATA_DIR / "messages.json"
TICKETS_FILE = DATA_DIR / "tickets.json"
KNOWLEDGE_FILE = DATA_DIR / "knowledge.json"

# ---------------------------
# Funções utilitárias
# ---------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json(path: Path, data: Dict[str, Any]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

# ---------------------------
# Tools principais
# ---------------------------
def fetch_webpage_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch the first URL and return content + metadata."""
    urls = payload.get("urls", [])
    if not urls:
        return {"error": "No URLs provided."}

    url = urls[0]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = re.sub(r"\s+", " ", soup.get_text(separator=" \n ")).strip()
        return {"content": text, "url": url, "title": title}
    except Exception as e:
        return {"error": str(e)}

def semantic_search_tool(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """TF-IDF semantic search."""
    query = payload.get("query", "")
    documents = payload.get("documents", [])
    top_k = int(payload.get("top_k", 5))

    if not documents:
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
        doc_vecs = vectorizer.fit_transform(documents)
        qv = vectorizer.transform([query])
        scores = (doc_vecs @ qv.T).toarray()[:, 0]
        idxs = np.argsort(-scores)
        return [{"index": int(i), "score": float(scores[i])} for i in idxs[:top_k]]
    except Exception as e:
        return [{"error": str(e)}]

def get_knowledge_tool(_: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return stored knowledge (url -> summary)."""
    data = _read_json(KNOWLEDGE_FILE)
    return data if isinstance(data, dict) else {}

def add_knowledge_url_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch and summarize a webpage, then add to knowledge.json."""
    url = payload.get("url")
    if not url:
        return {"error": "Missing 'url'."}

    res = fetch_webpage_tool({"urls": [url]})
    content = res.get("content", "")
    sents = re.split(r'(?<=[.!?])\s+', content)
    summary = " ".join(sents[:3]).strip()

    data = _read_json(KNOWLEDGE_FILE)
    data[url] = summary
    _write_json(KNOWLEDGE_FILE, data)
    return {url: summary}

def get_user_profile_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = payload.get("user_id")
    data = _read_json(USERS_FILE)
    for u in data.get("users", []):
        if u.get("user_id") == user_id:
            return u
    return {"error": "User not found."}

def create_support_ticket_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = payload.get("user_id")
    subject = payload.get("subject")
    body = payload.get("body")

    if not all([user_id, subject, body]):
        return {"error": "Missing fields (user_id, subject, body)"}

    tickets_data = _read_json(TICKETS_FILE)
    tickets = tickets_data.get("tickets", [])
    ticket_id = f"T{len(tickets)+1:06d}"
    ticket = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "subject": subject,
        "body": body,
        "status": "open"
    }
    tickets.append(ticket)
    tickets_data["tickets"] = tickets
    _write_json(TICKETS_FILE, tickets_data)
    return ticket

# ---------------------------
# Registro de tools MCP
# ---------------------------
MCP_TOOLS = {
    "fetch_webpage": fetch_webpage_tool,
    "semantic_search": semantic_search_tool,
    "get_knowledge": get_knowledge_tool,
    "add_knowledge_url": add_knowledge_url_tool,
    "get_user_profile": get_user_profile_tool,
    "create_support_ticket": create_support_ticket_tool,
}

# =====================================================
# TOOL 1: gemini_generate
# =====================================================
async def tool_gemini_generate(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates text using the local GeminiAgent wrapper.
    Keeps compatibility with the previous tool shape by returning a
    `result` dict containing `candidates` -> content -> parts -> text.
    Expects: { "prompt": str, "max_tokens": int }
    """
    prompt = params.get("prompt", "")
    max_tokens = params.get("max_tokens", 300)

    try:
        agent = GeminiAgent()

        # GeminiAgent.generate is synchronous; run it in a thread to avoid blocking
        resp = await asyncio.to_thread(agent.generate, prompt)

        # Normalize response into the GEMINI API-like structure expected by callers
        text = None
        if isinstance(resp, dict):
            text = resp.get("answer") or resp.get("text") or str(resp)
        else:
            text = str(resp)

        data = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        return {"tool_id": "gemini_generate", "result": data}
    except Exception as e:
        return {"error": str(e)}


# =====================================================
# TOOL 2: get_knowledge
# =====================================================
async def tool_get_knowledge(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves an answer from the KnowledgeAgent.
    Expects: { "query": str }
    """
    query = params.get("query")
    if not query:
        return {"error": "Faltando parâmetro 'query'."}

    agent = KnowledgeAgent()
    result = await agent.answer(query)
    return {"tool_id": "get_knowledge", "result": result}


# =====================================================
# TOOL 3: route_message
# =====================================================
async def tool_route_message(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Routes a message to the correct agent based on domain rules.
    Expects: { "message": str, "knowledge_context": str }
    """
    message = params.get("message")
    knowledge_context = params.get("knowledge_context", "(no knowledge entries)")

    if not message:
        return {"error": "Faltando parâmetro 'message'."}

    prompt = f"""
You are a ROUTING AI specialized in the CloudWalk and InfinityPay ecosystem.
Your task is to decide which specialized agent should handle a user message.

Respond ONLY with valid JSON:
{{
  "selected_agent": "SUPPORT" | "RETRIEVE" | "DIRECT" | "ADD_KNOWLEDGE",
  "is_final": "indicates if this is the final step"
  "reasoning": "short reasoning for your choice"
}}

## Domain context:
CloudWalk and InfinityPay provide payment solutions, APIs, financial platforms, and integrations for businesses.
User questions may involve topics such as: onboarding, fees, terminals, API usage, integrations, account setup, support requests, product updates, or business inquiries.

## Decision rules:
- Use **"RETRIEVE"** if the question asks about any information related to CloudWalk or InfinityPay — such as product details, pricing, API documentation, terminal usage, company policies, or other factual data — that could exist in the stored knowledge base.  
  -> Example: “How do I integrate InfinityPay API?”, “What are CloudWalk’s fees?”, “How does the terminal update work?”
- Use **"SUPPORT"** if the message is about **technical problems, bugs, access errors, or operational issues** with the system.  
  -> Example: “My terminal isn’t connecting”, “The dashboard is not loading”.
- Use **"ADD_KNOWLEDGE"** if the user wants to **add, update, or upload** new information to the knowledge base.  
  -> Example: “Add a new entry about API version 2.0 release.”
- Use **"DIRECT"** if the message is **personal, conversational, or unrelated to CloudWalk/InfinityPay**, and does not require retrieval or system support.  
  -> Example: “How are you?”, “Tell me a joke.”

Note:
If the user’s question could be improved or clarified by referencing company information, documentation, or internal resources, prefer **RETRIEVE** — since additional context can be acquired from stored knowledge.

Knowledge (summary):
{knowledge_context}

User message:
{message}
"""

    gemini = GeminiAgent()
    result = gemini.generate_structured(prompt, LLMDecision)

    if isinstance(result, BaseModel):
        return {"tool_id": "route_message", "result": result.dict()}
    return {"tool_id": "route_message", "result": result}


# =====================================================
# TOOL REGISTRY (para o MCP Object)
# =====================================================
TOOLS = {
    "gemini_generate": tool_gemini_generate,
    "get_knowledge": tool_get_knowledge,
    "route_message": tool_route_message,
}