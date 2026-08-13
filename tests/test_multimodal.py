import os
import pytest
from fastapi.testclient import TestClient
from apexrag.api.main import app
from apexrag.ingestion.multimodal_parser import MultimodalDocumentParser

client = TestClient(app)

def test_multimodal_parser_markdown():
    parser = MultimodalDocumentParser()
    sample_md = """# Architecture Overview
This is system documentation.

| Component | Status | Latency |
| --- | --- | --- |
| BM25 | Active | 10ms |
| ChromaDB | Active | 15ms |

```python
def retrieve():
    return "hybrid_results"
```

```mermaid
graph TD
    A --> B
```
"""
    chunks = parser._parse_text_markdown(sample_md, "test_doc.md")
    assert len(chunks) >= 3
    types = [c.chunk_type for c in chunks]
    assert "table" in types
    assert "code_snippet" in types
    assert "diagram" in types

def test_file_upload_api_endpoint():
    sample_content = b"""# ApexRAG Multimodal Specification
Table Matrix:

| Metric | Target |
| --- | --- |
| Faithfulness | 100% |

```python
print('ApexRAG Active')
```
"""
    response = client.post(
        "/api/ingest/file",
        files={"file": ("test_spec.md", sample_content, "text/markdown")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_spec.md"
    assert data["parsed_chunks_count"] >= 2
