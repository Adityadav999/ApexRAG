from typing import List, Dict, Any, Optional
from apexrag.retrieval.sparse import SparseRetriever, DocumentChunk
from apexrag.retrieval.dense import DenseRetriever

class HybridRetriever:
    """Hybrid Retriever combining BM25 Sparse & ChromaDB Dense retrieval via Reciprocal Rank Fusion (RRF)."""
    
    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60
    ):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        
    def add_documents(self, chunks: List[DocumentChunk]):
        """Index chunks into both sparse and dense stores."""
        self.sparse_retriever.add_documents(chunks)
        self.dense_retriever.add_documents(chunks)

    def search(
        self,
        query: str,
        top_k_sparse: int = 10,
        top_k_dense: int = 10,
        top_k_final: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Execute parallel BM25 and Dense vector search, then merge results using Reciprocal Rank Fusion (RRF).
        RRF_Score(d) = sum_{m in {bm25, dense}} ( 1 / (rrf_k + rank_m(d)) )
        """
        sparse_results = self.sparse_retriever.search(query, top_k=top_k_sparse)
        dense_results = self.dense_retriever.search(query, top_k=top_k_dense)
        
        chunk_registry: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}
        sparse_ranks: Dict[str, Optional[int]] = {}
        dense_ranks: Dict[str, Optional[int]] = {}
        
        # Process Sparse (BM25) Ranks
        for item in sparse_results:
            cid = item["chunk"]["id"]
            rank = item["sparse_rank"]
            chunk_registry[cid] = item["chunk"]
            sparse_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            
        # Process Dense (ChromaDB) Ranks
        for item in dense_results:
            cid = item["chunk"]["id"]
            rank = item["dense_rank"]
            chunk_registry[cid] = item["chunk"]
            dense_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            
        # Sort by merged RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        final_results = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_k_final], start=1):
            final_results.append({
                "chunk": chunk_registry[cid],
                "rrf_score": rrf_scores[cid],
                "rrf_rank": rank,
                "sparse_rank": sparse_ranks.get(cid),
                "dense_rank": dense_ranks.get(cid)
            })
            
        return final_results
