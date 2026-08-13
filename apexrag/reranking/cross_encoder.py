import math
from typing import List, Dict, Any, Optional

class CrossEncoderReranker:
    """Neural Cross-Encoder model to score and re-rank query-document pairs."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception:
            # Fallback to local heuristic re-ranker if torch/transformers isn't loaded
            self.model = None

    def rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Compute relevance scores for (query, document_text) pairs and return top_k reranked items.
        """
        if not items:
            return []

        pairs = [(query, item["chunk"]["text"]) for item in items]
        
        if self.model is not None:
            try:
                scores = self.model.predict(pairs)
                # Convert logits to probabilities via sigmoid if applicable
                prob_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in scores]
            except Exception:
                prob_scores = [self._heuristic_score(query, item["chunk"]["text"]) for item in items]
        else:
            prob_scores = [self._heuristic_score(query, item["chunk"]["text"]) for item in items]

        # Attach score and sort
        reranked_items = []
        for item, score in zip(items, prob_scores):
            copied = dict(item)
            copied["rerank_score"] = score
            reranked_items.append(copied)

        reranked_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        for rank, item in enumerate(reranked_items[:top_k], start=1):
            item["rerank_position"] = rank

        return reranked_items[:top_k]

    def _heuristic_score(self, query: str, text: str) -> float:
        """Heuristic fallback score based on exact keyword overlap & sequence length matching."""
        q_tokens = set(query.lower().split())
        t_tokens = text.lower().split()
        if not q_tokens or not t_tokens:
            return 0.0
        
        overlap_count = sum(1 for token in q_tokens if token in t_tokens)
        ratio = overlap_count / float(len(q_tokens))
        
        # Soft sigmoid mapping
        return 1.0 / (1.0 + math.exp(-3.0 * (ratio - 0.3)))
