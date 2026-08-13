import os
import json
from typing import List, Dict, Any, Optional
from apexrag.retrieval.sparse import DocumentChunk

class DenseRetriever:
    """Dense Semantic Search Engine backed by ChromaDB with InMemory numpy fallback."""

    def __init__(self, persist_dir: str, collection_name: str = "apexrag_chunks", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.chroma_client = None
        self.collection = None
        self.fallback_chunks: List[DocumentChunk] = []
        
        self._init_backend()

    def _init_backend(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
            except Exception:
                embedding_fn = None

            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn
            )
        except Exception:
            self.chroma_client = None
            self.collection = None
            self._load_fallback()

    def add_documents(self, chunks: List[DocumentChunk]):
        """Upsert document chunks into ChromaDB or fallback store."""
        if not chunks:
            return
            
        if self.collection is not None:
            ids = [c.id for c in chunks]
            documents = [c.text for c in chunks]
            metadatas = [c.metadata for c in chunks]
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        else:
            self.fallback_chunks.extend(chunks)
            self._save_fallback()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic similarity search against ChromaDB or fallback cosine matcher."""
        if self.collection is not None:
            if self.collection.count() == 0:
                return []
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            output = []
            if results and results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                documents = results["documents"][0] if results["documents"] else [""] * len(ids)
                metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
                distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
                
                for rank, (doc_id, doc_text, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
                    sim_score = max(0.0, 1.0 - float(dist))
                    output.append({
                        "chunk": {"id": doc_id, "text": doc_text, "metadata": meta},
                        "score": sim_score,
                        "dense_rank": rank,
                        "retriever_source": "chromadb"
                    })
            return output
        else:
            return self._search_fallback(query, top_k)

    def _search_fallback(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.fallback_chunks:
            return []
        
        q_tokens = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.fallback_chunks:
            c_tokens = chunk.text.lower().split()
            overlap = sum(1 for t in q_tokens if t in c_tokens)
            score = float(overlap) / max(1.0, float(len(q_tokens)))
            scored_chunks.append((chunk, score))
            
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        output = []
        for rank, (chunk, score) in enumerate(scored_chunks[:top_k], start=1):
            output.append({
                "chunk": chunk.model_dump(),
                "score": score,
                "dense_rank": rank,
                "retriever_source": "dense_fallback"
            })
        return output

    def _save_fallback(self):
        fallback_path = os.path.join(self.persist_dir, "dense_fallback.json")
        with open(fallback_path, "w") as f:
            json.dump([c.model_dump() for c in self.fallback_chunks], f)

    def _load_fallback(self):
        fallback_path = os.path.join(self.persist_dir, "dense_fallback.json")
        if os.path.exists(fallback_path):
            with open(fallback_path, "r") as f:
                data = json.load(f)
                self.fallback_chunks = [DocumentChunk(**c) for c in data]

    def count(self) -> int:
        if self.collection is not None:
            return self.collection.count()
        return len(self.fallback_chunks)

