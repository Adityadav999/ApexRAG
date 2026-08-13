import sys
import os
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apexrag.config import settings
from apexrag.retrieval.sparse import SparseRetriever, DocumentChunk
from apexrag.retrieval.dense import DenseRetriever
from apexrag.retrieval.hybrid import HybridRetriever

SAMPLE_DOCUMENTS = [
    DocumentChunk(
        id="doc-001",
        text="ApexRAG is an enterprise-grade Retrieval-Augmented Generation system. It implements Reciprocal Rank Fusion (RRF) combining sparse BM25 search with ChromaDB dense vector search to achieve high context precision.",
        metadata={"source": "ApexRAG_Whitepaper.pdf", "category": "Architecture"}
    ),
    DocumentChunk(
        id="doc-002",
        text="Cross-Encoder Reranking in ApexRAG uses the ms-marco-MiniLM-L-6-v2 model to evaluate top retrieved document chunks alongside user queries, pruning irrelevant context before feeding the generation step.",
        metadata={"source": "Reranking_Specs.md", "category": "ML Models"}
    ),
    DocumentChunk(
        id="doc-003",
        text="LangGraph powers ApexRAG's stateful self-correction loop. If retrieved context relevance drops below threshold 0.50, the workflow automatically expands and rewrites the query up to max_retries limit.",
        metadata={"source": "Orchestration_Guide.md", "category": "Workflow"}
    ),
    DocumentChunk(
        id="doc-004",
        text="Langfuse provides live tracing and telemetry for ApexRAG. It monitors P50, P90, and P95 latencies, tracks prompt and completion token counts, and calculates real-time USD costs per query.",
        metadata={"source": "Observability_Manual.pdf", "category": "Telemetry"}
    ),
    DocumentChunk(
        id="doc-005",
        text="Automated evaluation in ApexRAG uses the RAGAS framework to evaluate Faithfulness (0% target hallucination), Answer Relevance, Context Precision, and Context Recall across test datasets.",
        metadata={"source": "Evaluation_Benchmarks.md", "category": "Testing"}
    )
]

def seed_database():
    print("Seeding ApexRAG Knowledge Base with enterprise benchmark documents...")
    sparse = SparseRetriever(persist_path=settings.BM25_PERSIST_PATH)
    dense = DenseRetriever(persist_dir=settings.CHROMA_PERSIST_DIR, embedding_model=settings.EMBEDDING_MODEL_NAME)
    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=settings.RRF_K)

    hybrid.add_documents(SAMPLE_DOCUMENTS)
    print(f"[OK] Successfully indexed {len(SAMPLE_DOCUMENTS)} document chunks into BM25 and ChromaDB persistence stores.")


if __name__ == "__main__":
    seed_database()
