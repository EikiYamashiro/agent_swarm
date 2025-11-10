import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Type
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pydantic import BaseModel

def fetch_webpage(urls: List[str], query: str = "") -> Dict[str, Any]:
    """Fetch the first URL from the list and return extracted text and metadata.

    This is a small local replacement for an MCP fetch_webpage tool.
    """
    if not urls:
        return {"content": "", "url": "", "title": ""}
    url = urls[0]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator=" \n ")
        import re
        text = re.sub(r"\s+", " ", text).strip()
        return {"content": text, "url": url, "title": title}
    except Exception as e:
        raise e


def semantic_search(query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform a TF-IDF semantic search over the provided documents.

    Returns a list of dicts: {"index": int, "score": float}
    sorted by score descending.
    """
    if not documents:
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
        doc_vecs = vectorizer.fit_transform(documents)
        qv = vectorizer.transform([query])
        scores = (doc_vecs @ qv.T).toarray()[:, 0]
        idxs = np.argsort(-scores)
        results = []
        for i in idxs[:top_k]:
            results.append({"index": int(i), "score": float(scores[i])})
        return results
    except Exception as e:
        print(f"semantic_search error: {e}")
        return []


# gemini_agent wrapper: reuse existing GeminiAgent if available
class _LocalGeminiWrapper:
    def __init__(self):
        try:
            from .gemini_agent import GeminiAgent
            self._agent = GeminiAgent()
        except Exception:
            self._agent = None

    def generate(self, payload: Any) -> Dict[str, Any]:
        """If we have the real GeminiAgent, delegate to it.

        payload may be a string (prompt) or a dict expected by other code.
        """
        if self._agent:
            if isinstance(payload, str):
                return self._agent.generate(payload)
            # support dict with 'text' key like used in knowledge_agent_mcp
            if isinstance(payload, dict) and "text" in payload:
                return self._agent.generate(payload["text"])
        # Fallback: return a simple canned response using the payload
        text = ""
        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, dict):
            text = payload.get("text", "")
        # produce an abbreviated "answer" for fallback
        short = text[:800].strip()
        return {"text": f"[local-fallback] {short}"}

    def generate_structured(self, message: str, response_model: Type[BaseModel]) -> Dict[str, Any]:
        """Attempt to generate a structured response matching `response_model`.

        If the real GeminiAgent is available and exposes generate_structured, delegate to it.
        Otherwise, fall back to calling `generate` and attempt to parse JSON into the model.
        Returns a dict with keys: 'structured' (dict) or None, 'error' (str) or None, 'model' (optional).
        """
        # If underlying agent supports structured generation, use it
        if self._agent and hasattr(self._agent, "generate_structured"):
            try:
                return self._agent.generate_structured(message, response_model)
            except Exception as e:
                return {"structured": None, "error": str(e), "model": getattr(self._agent, "model", None)}

        # Fallback path: generate plain text and try to parse JSON into the response model
        resp = self.generate(message)
        raw = ""
        if isinstance(resp, dict):
            raw = resp.get("answer") or resp.get("text") or ""
        else:
            raw = str(resp)

        try:
            import json
            parsed = json.loads(raw.strip()) if raw else {}
        except Exception:
            parsed = {}

        # Try to validate/normalize using pydantic if response_model is provided
        try:
            if response_model is not None:
                validated = response_model(**parsed)
                return {"structured": validated.dict(), "error": None}
        except Exception as e:
            return {"structured": None, "error": f"validation_error: {e}", "raw": raw}

        return {"structured": parsed if isinstance(parsed, dict) else None, "error": None, "raw": raw}


gemini_agent = _LocalGeminiWrapper()

from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
MESSAGES_FILE = DATA_DIR / "messages.json"
TICKETS_FILE = DATA_DIR / "tickets.json"
KNOWLEDGE_FILE = DATA_DIR / "knowledge.json"

if not USERS_FILE.exists():
    sample_users = {
        "users": [
            {
                "user_id": "user_123",
                "name": "Alice Silva",
                "email": "alice@example.com",
                "account_status": "active",
                "created_at": "2024-01-15",
                "transactions": [
                    {"date": "2024-10-01", "type": "payment", "amount": "R$100.00"}
                ]
            },
            {
                "user_id": "user_456",
                "name": "Bruno Souza",
                "email": "bruno@example.com",
                "account_status": "active",
                "created_at": "2023-10-02"
            }
        ]
    }
    USERS_FILE.write_text(json.dumps(sample_users, indent=2, ensure_ascii=False), encoding="utf-8")

# Initialize messages and tickets files
if not MESSAGES_FILE.exists():
    MESSAGES_FILE.write_text(json.dumps({"messages": {}}, indent=2, ensure_ascii=False), encoding="utf-8")
if not TICKETS_FILE.exists():
    TICKETS_FILE.write_text(json.dumps({"tickets": []}, indent=2, ensure_ascii=False), encoding="utf-8")
if not KNOWLEDGE_FILE.exists():
    # initialize knowledge.json with INFINITEPAY_PAGES-like defaults if desired (empty by default)
    KNOWLEDGE_FILE.write_text(json.dumps({}, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]):
    # atomic write: write to a temp file and replace
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        tmp.replace(path)
    except Exception:
        # best-effort fallback
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_knowledge() -> Dict[str, str]:
    """Return the knowledge mapping (url -> summary) from knowledge.json."""
    data = _read_json(KNOWLEDGE_FILE)
    # Expecting a dict mapping url -> summary
    if isinstance(data, dict):
        return data
    return {}


def save_knowledge(knowledge: Dict[str, str]) -> None:
    # Read existing knowledge and merge to avoid accidental overwrite
    existing = _read_json(KNOWLEDGE_FILE)
    if not isinstance(existing, dict):
        existing = {}
    existing.update(knowledge)
    # create a backup copy before writing (timestamped) to allow recovery if needed
    try:
        import shutil, time
        backup = KNOWLEDGE_FILE.with_suffix(KNOWLEDGE_FILE.suffix + f".bak.{int(time.time())}")
        if KNOWLEDGE_FILE.exists():
            shutil.copy2(KNOWLEDGE_FILE, backup)
    except Exception:
        pass
    _write_json(KNOWLEDGE_FILE, existing)


def add_knowledge_url(url: str, max_sentences: int = 3) -> Dict[str, Any]:
    """Fetch the given URL, summarize its content and add it to knowledge.json.

    Returns the new entry as {url: summary}.
    """
    # reuse fetch_webpage
    res = fetch_webpage(urls=[url], query="Extract main content")
    content = res.get("content", "") if isinstance(res, dict) else ""
    # simple extractive summary: first N sentences
    import re
    def _summarize(text: str) -> str:
        sents = re.split(r'(?<=[.!?])\\s+', text)
        sents = [s.strip() for s in sents if s.strip()]
        if not sents:
            return text.strip()[:500]
        return " ".join(sents[:max_sentences])

    summary = _summarize(content)

    knowledge = get_knowledge()
    knowledge[url] = summary
    save_knowledge(knowledge)
    return {url: summary}


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    data = _read_json(USERS_FILE)
    for u in data.get("users", []):
        if u.get("user_id") == user_id:
            return u
    return None


def save_user_message(user_id: str, message: str, role: str = "user") -> None:
    data = _read_json(MESSAGES_FILE)
    msgs = data.get("messages", {})
    user_msgs = msgs.get(user_id, [])
    user_msgs.append({"role": role, "message": message})
    msgs[user_id] = user_msgs
    data["messages"] = msgs
    _write_json(MESSAGES_FILE, data)


def get_user_messages(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    data = _read_json(MESSAGES_FILE)
    msgs = data.get("messages", {}).get(user_id, [])
    return msgs[-limit:]


def create_support_ticket(user_id: str, subject: str, body: str) -> Dict[str, Any]:
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


def list_tickets_for_user(user_id: str) -> List[Dict[str, Any]]:
    tickets_data = _read_json(TICKETS_FILE)
    return [t for t in tickets_data.get("tickets", []) if t.get("user_id") == user_id]