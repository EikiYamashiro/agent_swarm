from pydantic import BaseModel
from typing import Dict, Any, Optional


class LLMDecision(BaseModel):
    selected_agent: str
    is_final: bool
    reasoning: Optional[str] = None