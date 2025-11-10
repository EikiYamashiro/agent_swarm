from google import genai
import os
import asyncio
import time
from typing import Dict, Any, Type
from pydantic import BaseModel, ValidationError
import json
import re
import random


class GeminiAgent:
    """Lightweight Gemini wrapper with retry and structured generation."""

    def __init__(self, max_retries: int = 5, base_delay: float = 2.0):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.max_retries = max_retries
        self.base_delay = base_delay  # seconds

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

    def _retry_logic(self, attempt: int) -> None:
        """Wait with exponential backoff + jitter before retrying."""
        delay = self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
        print(f"⚠️ Tentativa {attempt} falhou, aguardando {delay:.2f}s antes de tentar novamente...")
        time.sleep(delay)

    def generate(self, message: str) -> Dict[str, Any]:
        """Generate a raw text response from Gemini with retry support."""
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1
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
                if "429" in str(e):
                    if attempt < self.max_retries:
                        self._retry_logic(attempt)
                        continue
                    else:
                        return {
                            "error": f"Too many requests (429) after {self.max_retries} retries.",
                            "model": self.model,
                        }
                else:
                    return {
                        "answer": f"Error generating response: {str(e)}",
                        "model": self.model,
                        "error": str(e),
                    }

            finally:
                if client is not None:
                    self._close_client_async(client)

        return {"error": "Max retries exceeded.", "model": self.model}

    def generate_structured(self, message: str, response_model: Type[BaseModel]):
        """Generate a structured (typed) response using Gemini with retry support."""
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1
            client = None
            try:
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model,
                    contents=message,
                )
                text = getattr(response, "text", None) or str(response)

                cleaned = (
                    text.strip()
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

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
                if "429" in str(e):
                    if attempt < self.max_retries:
                        self._retry_logic(attempt)
                        continue
                    else:
                        return {
                            "error": f"Too many requests (429) after {self.max_retries} retries.",
                            "model": self.model,
                        }
                else:
                    return {"error": str(e), "model": self.model}

            finally:
                if client is not None:
                    self._close_client_async(client)

        return {"error": "Max retries exceeded.", "model": self.model}
