import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from apexrag.config import settings
from apexrag.retrieval.sparse import SparseRetriever, DocumentChunk
from apexrag.retrieval.dense import DenseRetriever
from apexrag.retrieval.hybrid import HybridRetriever
from apexrag.reranking.cross_encoder import CrossEncoderReranker
from apexrag.graph.workflow import build_apexrag_graph
from apexrag.observability.tracer import ApexTracer, MetricsRepository
from apexrag.eval.evaluator import RagasEvaluator, EvaluationSample
from apexrag.ingestion.multimodal_parser import MultimodalDocumentParser

# Resolve the path to index.html at the project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_INDEX_HTML = os.path.join(_PROJECT_ROOT, "index.html")

app = FastAPI(
    title="ApexRAG Engine API",
    description="Enterprise-Grade RAG System with Multimodal Document Ingestion.",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Singletons
sparse_retriever = SparseRetriever(persist_path=settings.BM25_PERSIST_PATH)
sparse_retriever.load()

dense_retriever = DenseRetriever(
    persist_dir=settings.CHROMA_PERSIST_DIR,
    embedding_model=settings.EMBEDDING_MODEL_NAME
)

hybrid_retriever = HybridRetriever(
    sparse_retriever=sparse_retriever,
    dense_retriever=dense_retriever,
    rrf_k=settings.RRF_K
)

reranker = CrossEncoderReranker(model_name=settings.RERANKER_MODEL_NAME)
evaluator = RagasEvaluator()
multimodal_parser = MultimodalDocumentParser()

# Models
class QueryRequest(BaseModel):
    query: str
    max_loops: Optional[int] = settings.MAX_SELF_CORRECTION_LOOPS

class IngestDocument(BaseModel):
    id: str
    text: str
    filename: Optional[str] = "manual_entry.txt"
    chunk_type: Optional[str] = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchIngestRequest(BaseModel):
    documents: List[IngestDocument]

class EvaluateRequest(BaseModel):
    samples: List[EvaluationSample]


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "bm25_chunks": len(sparse_retriever.chunks),
        "chroma_chunks": dense_retriever.count(),
        "storage_engine": "ChromaDB + SQLite (Capacity: >100GB / Millions of Chunks)",
        "max_file_size_mb": 250
    }


@app.get("/api/documents")
def get_documents():
    file_map: Dict[str, Dict[str, Any]] = {}
    for c in sparse_retriever.chunks:
        fname = c.metadata.get("source", "document.txt")
        ctype = c.metadata.get("chunk_type", "text")
        if fname not in file_map:
            file_map[fname] = {
                "filename": fname,
                "total_chunks": 0,
                "types": {},
                "sample_snippet": c.text[:140] + ("..." if len(c.text) > 140 else "")
            }
        file_map[fname]["total_chunks"] += 1
        file_map[fname]["types"][ctype] = file_map[fname]["types"].get(ctype, 0) + 1
    return list(file_map.values())


@app.post("/api/query")
def execute_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    tracer = ApexTracer(trace_name="ApexRAG_Graph_Execution")
    graph = build_apexrag_graph(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        tracer=tracer
    )

    initial_state = {
        "query": req.query,
        "original_query": req.query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "relevance_score": 0.0,
        "is_relevant": False,
        "loop_count": 0,
        "generation": "",
        "citations": [],
        "execution_trace": {},
        "prompt_tokens": 0,
        "completion_tokens": 0
    }

    final_state = graph.invoke(initial_state)
    telemetry = tracer.finish(
        query=req.query,
        prompt_tokens=final_state.get("prompt_tokens", 0),
        completion_tokens=final_state.get("completion_tokens", 0),
        loop_count=final_state.get("loop_count", 0),
        relevance_score=final_state.get("relevance_score", 0.0)
    )

    return {
        "query": req.query,
        "answer": final_state.get("generation", ""),
        "citations": final_state.get("citations", []),
        "self_correction_loops": final_state.get("loop_count", 0),
        "relevance_score": final_state.get("relevance_score", 0.0),
        "retrieved_chunks_count": len(final_state.get("retrieved_docs", [])),
        "reranked_chunks": final_state.get("reranked_docs", []),
        "telemetry": telemetry.model_dump()
    }


@app.post("/api/ingest")
def ingest_documents(req: BatchIngestRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided for ingestion.")

    chunks = []
    for doc in req.documents:
        meta = doc.metadata or {}
        meta["source"] = doc.filename
        meta["chunk_type"] = doc.chunk_type or "text"
        chunks.append(DocumentChunk(id=doc.id, text=doc.text, metadata=meta))

    hybrid_retriever.add_documents(chunks)
    return {
        "status": "success",
        "ingested_count": len(chunks),
        "total_bm25_chunks": len(sparse_retriever.chunks),
        "total_chroma_chunks": dense_retriever.count()
    }


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    filename = file.filename or "uploaded_document.txt"

    parsed_chunks = multimodal_parser.parse_file(file_bytes, filename)
    if not parsed_chunks:
        raise HTTPException(status_code=400, detail=f"Could not extract content from file '{filename}'.")

    doc_chunks = []
    for pchunk in parsed_chunks:
        meta = pchunk.metadata or {}
        meta["source"] = filename
        meta["chunk_type"] = pchunk.chunk_type
        doc_chunks.append(DocumentChunk(id=pchunk.id, text=pchunk.text, metadata=meta))

    hybrid_retriever.add_documents(doc_chunks)
    return {
        "status": "success",
        "filename": filename,
        "parsed_chunks_count": len(doc_chunks),
        "chunk_types_breakdown": {
            "text": sum(1 for c in doc_chunks if c.metadata.get("chunk_type") == "text"),
            "table": sum(1 for c in doc_chunks if c.metadata.get("chunk_type") == "table"),
            "code_snippet": sum(1 for c in doc_chunks if c.metadata.get("chunk_type") == "code_snippet"),
            "diagram": sum(1 for c in doc_chunks if c.metadata.get("chunk_type") == "diagram"),
            "image_caption": sum(1 for c in doc_chunks if c.metadata.get("chunk_type") == "image_caption")
        },
        "total_bm25_chunks": len(sparse_retriever.chunks),
        "total_chroma_chunks": dense_retriever.count()
    }


@app.get("/api/metrics")
def get_metrics():
    return MetricsRepository.get_summary()


@app.post("/api/evaluate")
def run_evaluation(req: EvaluateRequest):
    report = evaluator.evaluate_samples(req.samples)
    return report.model_dump()


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Read index.html from disk on every request so changes take effect without restart."""
    with open(_INDEX_HTML, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
