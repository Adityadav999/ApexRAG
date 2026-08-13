import os
import pickle
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

def simple_tokenize(text: str) -> List[str]:
    """Clean and tokenize text into lowercase word tokens."""
    return re.findall(r'\w+', text.lower())

class SparseRetriever:
    """BM25 Sparse Keyword Search Engine."""
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self.chunks: List[DocumentChunk] = []
        self.bm25 = None
        
    def add_documents(self, chunks: List[DocumentChunk]):
        """Index a list of DocumentChunks into BM25."""
        if not chunks:
            return
            
        from rank_bm25 import BM25Okapi
        self.chunks.extend(chunks)
        tokenized_corpus = [simple_tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        if self.persist_path:
            self.save()
            
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search BM25 index and return sorted list of chunk dicts with scores."""
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []
            
        scores = self.bm25.get_scores(tokenized_query)
        # Zip chunks with scores
        scored_chunks = list(zip(self.chunks, scores))
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (chunk, score) in enumerate(scored_chunks[:top_k], start=1):
            results.append({
                "chunk": chunk.model_dump(),
                "score": float(score),
                "sparse_rank": rank,
                "retriever_source": "bm25"
            })
        return results
        
    def save(self):
        """Persist BM25 index and document chunks to file."""
        if not self.persist_path:
            return
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump({"chunks": [c.model_dump() for c in self.chunks]}, f)
            
    def load(self):
        """Load BM25 index from file if exists."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return False
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
            chunks_data = data.get("chunks", [])
            self.chunks = [DocumentChunk(**c) for c in chunks_data]
            if self.chunks:
                from rank_bm25 import BM25Okapi
                tokenized_corpus = [simple_tokenize(c.text) for c in self.chunks]
                self.bm25 = BM25Okapi(tokenized_corpus)
        return True
