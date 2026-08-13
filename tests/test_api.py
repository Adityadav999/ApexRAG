from fastapi.testclient import TestClient
from apexrag.api.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"

def test_api_ingest_and_query():
    # Ingest document
    ingest_res = client.post("/api/ingest", json={
        "documents": [
            {
                "id": "test-doc-1",
                "filename": "TestSpec.pdf",
                "text": "ApexRAG system provides P50 and P95 latency tracing using Langfuse observability integration."
            }
        ]
    })
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"

    # Query endpoint
    query_res = client.post("/api/query", json={
        "query": "What telemetry does ApexRAG provide?"
    })
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "answer" in q_data
    assert "telemetry" in q_data
    assert q_data["telemetry"]["total_latency_ms"] >= 0

def test_api_metrics():
    res = client.get("/api/metrics")
    assert res.status_code == 200
    assert "p50_latency_ms" in res.json()

def test_api_evaluate():
    res = client.post("/api/evaluate", json={
        "samples": [
            {
                "user_input": "What is ApexRAG?",
                "response": "ApexRAG is an enterprise RAG system.",
                "retrieved_contexts": ["ApexRAG is an enterprise RAG system with hybrid retrieval."]
            }
        ]
    })
    assert res.status_code == 200
    data = res.json()
    assert "faithfulness_score" in data
    assert data["faithfulness_score"] > 0
