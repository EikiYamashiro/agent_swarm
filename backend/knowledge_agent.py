from typing import List, Dict, Any, Tuple
from fastapi import HTTPException
from .mcp_client import MCPClient
import asyncio


class KnowledgeAgent:

    def __init__(self):
        # docs will be list of (text_chunk, source_url)
        self.docs = []
        self._indexed = False
        

    def _fetch_text(self, url: str) -> str:
        """Fetch and extract clean text from a webpage using MCP fetch_webpage."""
        try:
            from .mcpo_tools import fetch_webpage_tool
            result = fetch_webpage_tool({"urls": [url]})
            if not result or not result.get("content"):
                return ""
            return result["content"]
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    def _chunk_text(self, text: str, max_chars: int = 800) -> List[str]:
        """Split text into chunks around sentence boundaries."""
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chars and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            
        return chunks

    def build_index(self):
        """Download pages and build index using MCP fetch_webpage."""
        if self._indexed:
            return

        # Load known URLs from knowledge.json via MCP tools
        try:
            from .mcpo_tools import get_knowledge_tool
            knowledge = get_knowledge_tool()
        except Exception:
            knowledge = {}

        all_chunks = []
        for url, stored_summary in knowledge.items():
            text = self._fetch_text(url)
            # if fetch fails, fall back to stored summary
            if not text:
                text = stored_summary or ""
            if not text:
                continue

            # Chunk the fetched text (or stored summary when fetch failed)
            chunks = self._chunk_text(text)
            for c in chunks:
                all_chunks.append((c, url))

            # Also include the stored summary as an additional chunk if present and not duplicate
            try:
                if stored_summary and stored_summary.strip():
                    # avoid adding identical chunk twice
                    if stored_summary not in chunks:
                        all_chunks.append((stored_summary, url))
            except Exception:
                pass

        if not all_chunks:
            self.docs = []
            self._indexed = True
            return

        self.docs = all_chunks
        self._indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Return top_k (text, source_url, score) tuples relevant to query using MCP semantic search."""
        if not self._indexed:
            self.build_index()
        if not self.docs:
            return []

        try:
            from .mcpo_tools import semantic_search_tool
            texts = [text for text, _ in self.docs]
            
            payload = {
                "query": query,
                "documents": texts,
                "top_k": top_k
            }
            results = semantic_search_tool(payload)
            if not results:
                return []

            hits = []
            for res in results:
                if "error" in res:
                    print(f"Error in search result: {res['error']}")
                    continue
                idx = int(res.get("index", -1))
                if idx < 0 or idx >= len(texts):
                    continue
                text = texts[idx]
                url = next((url for t, url in self.docs if t == text), "")
                score = float(res.get("score", 0.0))
                hits.append((text, url, score))

            return hits

        except Exception as e:
            print(f"Error in retrieve: {e}")
            return []

    async def answer(self, query: str) -> Dict[str, Any]:
        """Answer a question using retrieved passages via MCP Gemini Flash."""
        try:
            hits = self.retrieve(query, top_k=3)
            if not hits:
                return {
                    "answer": "Não encontrei informações suficientes para responder.",
                    "sources": []
                }

            texts = []
            sources = set()
            for text, url, score in hits:
                if score < 0.1:
                    continue
                texts.append(text)
                sources.add(url)

            if not texts:
                return {"answer": "Nenhuma informação relevante encontrada.", "sources": []}

            context = "\n\n".join(texts)
            prompt = (
                f"Com base neste contexto, responda a pergunta abaixo de forma direta e técnica:\n\n"
                f"Pergunta: {query}\n\n"
                f"Contexto:\n{context}"
            )

            from .mcpo_tools import tool_gemini_generate
            
            payload = {
                "prompt": prompt,
                "max_tokens": 300
            }
            
            response = await tool_gemini_generate(payload)
            
            answer = None
            if isinstance(response, dict):
                result = response.get("result", {})
                if isinstance(result, dict):
                    candidates = result.get("candidates")
                    if candidates:
                        answer = candidates[0]["content"]["parts"][0]["text"]

            if not answer:
                answer = "Não foi possível gerar uma resposta."

            return {"answer": answer, "sources": list(sources)}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta: {str(e)}")