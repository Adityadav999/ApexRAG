# ApexRAG Enterprise System

ApexRAG is a production-ready Retrieval-Augmented Generation engine designed for zero-hallucination accuracy, stateful self-correction, real-time observability, and automated evaluation metrics.

```
[ User Query ]
       │
       ▼
[ Orchestration (LangGraph State Machine) ]
       │
       ├─► 1. Hybrid Retrieval (BM25 Keyword + ChromaDB Vector Search)
       │
       ├─► 2. Cross-Encoder Reranking (Top-K relevance filtering via SentenceTransformers)
       │
       ├─► 3. Self-Correction Loop (Relevance check → Query Expansion if score is low)
       │
       ├─► 4. Generation & Citation Enforcement (Extract precise response with sources)
       │
       ▼
[ Parallel Observability & Evaluation ]
       ├─► Live Tracing & Telemetry: Langfuse (P50/P95 latency, costs, token usage)
       └─► Offline Benchmarking: RAGAS (Faithfulness, Context Precision, Answer Relevance)
```

## Features

- **Advanced Hybrid Retrieval**: Combines BM25 sparse keyword search and ChromaDB dense vector search using **Reciprocal Rank Fusion (RRF)** ($k=60$).
- **Neural Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` to evaluate top candidate chunks and eliminate irrelevant context before LLM prompt assembly.
- **Stateful Self-Correction (LangGraph)**: Evaluates context quality with automated query rewriting and expansion loops if initial relevance scores fall below threshold.
- **Live Telemetry & Observability (Langfuse)**: Captures trace trees, measures P50, P90, and P95 latencies, tracks prompt/completion token usage, and calculates real-time USD costs per query.
- **Automated Evaluation Pipeline (RAGAS)**: Benchmarks Faithfulness (0% hallucination target), Answer Relevance, Context Precision, and Context Recall.
- **Interactive Glassmorphic Web Dashboard**: Includes a live query workbench, citation inspector, telemetry monitor, and knowledge base manager.

---

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Seed Knowledge Base
```bash
python scripts/seed_data.py
```

### 3. Launch ApexRAG Server & Web Dashboard
```bash
python -m uvicorn apexrag.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at `http://localhost:8000` to access the interactive web dashboard.

---

## Testing & Benchmarks

Run full automated test suite for Hybrid Retrieval, LangGraph workflow, and FastAPI endpoints:
```bash
python -m pytest tests/
```

---

## Configuration

Settings can be customized in `apexrag/config.py` or `.env`:
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-api-key-here
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```
