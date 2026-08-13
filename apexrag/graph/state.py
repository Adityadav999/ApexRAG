from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class ApexRAGState(TypedDict):
    query: str
    original_query: str
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    relevance_score: float
    is_relevant: bool
    loop_count: int
    generation: str
    citations: List[Dict[str, Any]]
    execution_trace: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
