import os
import shutil
import pytest
from apexrag.retrieval.sparse import SparseRetriever, DocumentChunk
from apexrag.retrieval.dense import DenseRetriever
from apexrag.retrieval.hybrid import HybridRetriever
from apexrag.reranking.cross_encoder import CrossEncoderReranker

TEST_DIR = os.path.join(os.getcwd(), "data", "test_tmp")

@pytest.fixture(autouse=True)
def cleanup():
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)

def test_sparse_bm25_search():
    sparse = SparseRetriever(persist_path=os.path.join(TEST_DIR, "bm25.pkl"))
    chunks = [
        DocumentChunk(id="1", text="Kubernetes cluster orchestration and container deployment.", metadata={"source": "k8s.md"}),
        DocumentChunk(id="2", text="ChromaDB dense vector database embeddings.", metadata={"source": "db.md"})
    ]
    sparse.add_documents(chunks)
    
    results = sparse.search("Kubernetes cluster", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk"]["id"] == "1"

def test_hybrid_rrf_fusion():
    sparse = SparseRetriever(persist_path=os.path.join(TEST_DIR, "bm25.pkl"))
    dense = DenseRetriever(persist_dir=os.path.join(TEST_DIR, "chroma"))
    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=60)
    
    chunks = [
        DocumentChunk(id="c1", text="LangGraph stateful self-correction loops for RAG.", metadata={"source": "graph.md"}),
        DocumentChunk(id="c2", text="RAGAS evaluation metrics for faithfulness and precision.", metadata={"source": "eval.md"})
    ]
    hybrid.add_documents(chunks)
    
    results = hybrid.search("stateful self-correction", top_k_final=2)
    assert len(results) > 0
    assert "rrf_score" in results[0]

def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker()
    items = [
        {"chunk": {"id": "1", "text": "Unrelated topic on cooking pasta."}},
        {"chunk": {"id": "2", "text": "Langfuse live tracing monitors latency and cost."}}
    ]
    reranked = reranker.rerank("What is Langfuse used for?", items, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk"]["id"] == "2"
