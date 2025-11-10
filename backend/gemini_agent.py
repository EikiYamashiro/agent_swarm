from google import genai
import os
import asyncio
from typing import Dict, Any, Type
from pydantic import BaseModel, ValidationError
import json
import re

class GeminiAgent:
    """Lightweight Gemini wrapper with text and structured generation."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

    def _close_client_async(self, client: Any) -> None:
        """Attempt to close the client cleanly."""
        try:
            aclose = getattr(client, "aclose", None)
            close = getattr(client, "close", None)
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if aclose and loop is not None:
                loop.create_task(aclose())
                return

            if close:
                close()
        except Exception:
            pass
        
    def generate(self, message: str) -> Dict[str, Any]:
        """Generate a raw text response from Gemini."""
        client = None
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=message,
            )
            text = getattr(response, "text", None) or str(response)
            return {"answer": text, "text": text, "model": self.model}
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "model": self.model,
                "error": str(e),
            }
        finally:
            if client is not None:
                self._close_client_async(client)
            
    def generate_structured(self, message: str, response_model: Type[BaseModel]):
        """
        Generate a structured (typed) response using Gemini.
        """
        client = None
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=message,
            )
            
            text = getattr(response, "text", None) or str(response)

            # Limpeza do texto
            cleaned = (
                text.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            # Corrige True/False para JSON válido
            normalized = re.sub(r'\bFalse\b', 'false', cleaned)
            normalized = re.sub(r'\bTrue\b', 'true', normalized)

            try:
                data = json.loads(normalized)
                return response_model(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                return {
                    "error": f"Invalid model output: {e}",
                    "raw_text": text,
                    "normalized_text": normalized,
                }

        except Exception as e:
            return {
                "error": str(e),
                "model": self.model,
            }
        finally:
            if client is not None:
                self._close_client_async(client)