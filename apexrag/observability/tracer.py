import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from apexrag.config import settings

class ExecutionSpan(BaseModel):
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryTelemetry(BaseModel):
    trace_id: str
    query: str
    total_latency_ms: float
    spans: List[ExecutionSpan]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    loop_count: int = 0
    relevance_score: float = 0.0

class MetricsRepository:
    """In-memory telemetry store to compute P50, P90, P95 metrics."""
    _queries: List[QueryTelemetry] = []

    @classmethod
    def record(cls, telemetry: QueryTelemetry):
        cls._queries.append(telemetry)

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        if not cls._queries:
            return {
                "total_queries": 0,
                "p50_latency_ms": 0.0,
                "p90_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0.0,
                "average_relevance": 0.0
            }
        
        latencies = [q.total_latency_ms for q in cls._queries]
        costs = [q.estimated_cost_usd for q in cls._queries]
        relevances = [q.relevance_score for q in cls._queries]
        prompt_tokens = sum(q.prompt_tokens for q in cls._queries)
        completion_tokens = sum(q.completion_tokens for q in cls._queries)

        return {
            "total_queries": len(cls._queries),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p90_latency_ms": round(float(np.percentile(latencies, 90)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_cost_usd": round(sum(costs), 6),
            "average_relevance": round(float(np.mean(relevances)), 3),
            "recent_traces": [q.model_dump() for q in cls._queries[-10:]]
        }

class ApexTracer:
    """Telemetry & Langfuse tracer wrapper."""

    def __init__(self, trace_name: str = "ApexRAG_Execution"):
        self.trace_id = str(uuid.uuid4())
        self.trace_name = trace_name
        self.spans: List[ExecutionSpan] = []
        self.start_time = time.time()
        self.langfuse_client = None
        self.langfuse_trace = None
        
        if settings.ENABLE_LANGFUSE and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                from langfuse import Langfuse
                self.langfuse_client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
                self.langfuse_trace = self.langfuse_client.trace(
                    id=self.trace_id,
                    name=self.trace_name
                )
            except Exception:
                self.langfuse_client = None

    def start_span(self, span_name: str, metadata: Optional[Dict[str, Any]] = None) -> ExecutionSpan:
        span = ExecutionSpan(
            name=span_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self.spans.append(span)
        return span

    def end_span(self, span: ExecutionSpan, extra_meta: Optional[Dict[str, Any]] = None):
        span.end_time = time.time()
        span.duration_ms = round((span.end_time - span.start_time) * 1000.0, 2)
        if extra_meta:
            span.metadata.update(extra_meta)
            
        if self.langfuse_trace:
            try:
                self.langfuse_trace.span(
                    name=span.name,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    metadata=span.metadata
                )
            except Exception:
                pass

    def finish(
        self,
        query: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        loop_count: int = 0,
        relevance_score: float = 0.0
    ) -> QueryTelemetry:
        total_latency = round((time.time() - self.start_time) * 1000.0, 2)
        
        # Calculate cost (e.g. OpenAI GPT-4o-mini rates: $0.15/1M input, $0.60/1M output)
        cost_usd = (prompt_tokens * 0.15 / 1e6) + (completion_tokens * 0.60 / 1e6)

        telemetry = QueryTelemetry(
            trace_id=self.trace_id,
            query=query,
            total_latency_ms=total_latency,
            spans=self.spans,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=round(cost_usd, 6),
            loop_count=loop_count,
            relevance_score=round(relevance_score, 3)
        )
        
        MetricsRepository.record(telemetry)
        return telemetry
