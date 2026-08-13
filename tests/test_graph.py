import os
import shutil
import pytest
from apexrag.retrieval.sparse import SparseRetriever, DocumentChunk
from apexrag.retrieval.dense import DenseRetriever
from apexrag.retrieval.hybrid import HybridRetriever
from apexrag.reranking.cross_encoder import CrossEncoderReranker
from apexrag.graph.workflow import build_apexrag_graph

TEST_DIR = os.path.join(os.getcwd(), "data", "test_graph_tmp")

@pytest.fixture(autouse=True)
def cleanup():
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)

def test_langgraph_execution_flow():
    sparse = SparseRetriever(persist_path=os.path.join(TEST_DIR, "bm25.pkl"))
    dense = DenseRetriever(persist_dir=os.path.join(TEST_DIR, "chroma"))
    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)
    reranker = CrossEncoderReranker()

    chunks = [
        DocumentChunk(
            id="g1",
            text="ApexRAG integrates Reciprocal Rank Fusion (RRF) with Cross-Encoder reranking and LangGraph self-correction.",
            metadata={"source": "ApexRAG.md"}
        )
    ]
    hybrid.add_documents(chunks)

    graph = build_apexrag_graph(hybrid_retriever=hybrid, reranker=reranker)

    initial_state = {
        "query": "How does ApexRAG work?",
        "original_query": "How does ApexRAG work?",
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

    assert "generation" in final_state
    assert len(final_state["generation"]) > 0
    assert len(final_state["citations"]) > 0
    assert final_state["citations"][0]["source_id"] == "Source 1"
