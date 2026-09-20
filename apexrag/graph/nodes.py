import re
import math
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

from apexrag.config import settings
from apexrag.graph.state import ApexRAGState
from apexrag.retrieval.hybrid import HybridRetriever
from apexrag.reranking.cross_encoder import CrossEncoderReranker
from apexrag.observability.tracer import ApexTracer


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def _sentence_score(query_tokens: List[str], sentence: str) -> float:
    """Score a sentence by term overlap with the query (TF-IDF-style)."""
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return 0.0
    query_set = set(query_tokens)
    # Stop words to ignore
    stopwords = {
        'a', 'an', 'the', 'is', 'in', 'on', 'at', 'to', 'for', 'of', 'and',
        'or', 'but', 'with', 'from', 'by', 'as', 'be', 'are', 'was', 'were',
        'i', 'it', 'this', 'that', 'what', 'how', 'which', 'when', 'where',
        'do', 'does', 'did', 'have', 'has', 'had', 'will', 'would', 'can',
        'could', 'should', 'may', 'might', 'my', 'your', 'its', 'their'
    }
    meaningful_query = query_set - stopwords
    if not meaningful_query:
        meaningful_query = query_set

    matches = sum(1 for t in sent_tokens if t in meaningful_query)
    score = matches / (math.sqrt(len(sent_tokens)) + 1e-6)
    return score


def _extract_best_sentences(query: str, text: str, n: int = 3) -> str:
    """Extract the top-N most relevant sentences from a chunk for the given query."""
    query_tokens = _tokenize(query)
    # Split into sentences (handle line breaks too)
    raw_sents = re.split(r'(?<=[.!?])\s+|\n', text)
    sentences = [s.strip() for s in raw_sents if len(s.strip()) > 20]
    if not sentences:
        return text[:500]

    scored = [(s, _sentence_score(query_tokens, s)) for s in sentences]
    # Sort by score descending, keep original order for top-N
    top_indices = sorted(
        range(len(scored)),
        key=lambda i: scored[i][1],
        reverse=True
    )[:n]
    # Re-sort by original order for coherent reading
    top_indices = sorted(top_indices)
    return " ... ".join(sentences[i] for i in top_indices)


def _build_direct_answer(query: str, context_blocks: List[str], source_labels: List[str]) -> str:
    """
    Build a precise, direct answer from extracted evidence sentences.
    Groups evidence per source and states the answer clearly.
    """
    query_tokens = _tokenize(query)
    
    # Detect question type for targeted answer prefix
    q_lower = query.lower()
    if q_lower.startswith(('what is', 'what are', 'define', 'describe')):
        prefix = "Based on the indexed documents:"
    elif q_lower.startswith(('how', 'explain')):
        prefix = "According to the indexed documents:"
    elif q_lower.startswith(('list', 'enumerate', 'name')):
        prefix = "The following were found in the indexed documents:"
    elif q_lower.startswith(('when', 'where', 'who', 'which')):
        prefix = "From the indexed documents:"
    else:
        prefix = "Based on the retrieved context:"

    answer_parts = [prefix, ""]
    
    for i, (block, label) in enumerate(zip(context_blocks, source_labels), start=1):
        evidence = _extract_best_sentences(query, block, n=4)
        if evidence.strip():
            answer_parts.append(f"[{label}]")
            answer_parts.append(evidence.strip())
            answer_parts.append("")

    if len(answer_parts) <= 2:
        return "No sufficiently relevant context was found for this query. Try uploading more documents or rephrasing your question."

    answer_parts.append("---")
    answer_parts.append(
        f"Answer synthesised from {len(context_blocks)} source(s). "
        "Citations shown above correspond to the indexed document chunks."
    )
    return "\n".join(answer_parts)


class GraphNodes:
    """Encapsulates LangGraph state machine node operations."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        tracer: Optional[ApexTracer] = None
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.tracer = tracer

    def retrieve_node(self, state: ApexRAGState) -> Dict[str, Any]:
        """Node 1: Execute BM25 + ChromaDB Hybrid Search with RRF."""
        query = state["query"]
        span = None
        if self.tracer:
            span = self.tracer.start_span("Hybrid_Retrieval_RRF", {"query": query})

        retrieved = self.hybrid_retriever.search(
            query=query,
            top_k_sparse=settings.TOP_K_SPARSE,
            top_k_dense=settings.TOP_K_DENSE,
            top_k_final=settings.TOP_K_RRF
        )

        if self.tracer and span:
            self.tracer.end_span(span, {"count": len(retrieved)})

        return {"retrieved_docs": retrieved}

    def rerank_node(self, state: ApexRAGState) -> Dict[str, Any]:
        """Node 2: Cross-Encoder Reranking — prune to TOP_K_RERANK best chunks."""
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]

        span = None
        if self.tracer:
            span = self.tracer.start_span("Cross_Encoder_Rerank", {"input_count": len(retrieved_docs)})

        reranked = self.reranker.rerank(
            query=query,
            items=retrieved_docs,
            top_k=settings.TOP_K_RERANK
        )

        if self.tracer and span:
            self.tracer.end_span(span, {"output_count": len(reranked)})

        return {"reranked_docs": reranked}

    def grade_relevance_node(self, state: ApexRAGState) -> Dict[str, Any]:
        """Node 3: Grade relevance using max rerank score of top doc (not average)."""
        reranked_docs = state["reranked_docs"]

        span = None
        if self.tracer:
            span = self.tracer.start_span("Grade_Relevance")

        if not reranked_docs:
            top_score = 0.0
        else:
            # Use max score (the best doc) — more sensitive than average
            top_score = max(d.get("rerank_score", 0.0) for d in reranked_docs)

        is_relevant = top_score >= settings.RELEVANCE_THRESHOLD

        if self.tracer and span:
            self.tracer.end_span(span, {"top_score": top_score, "is_relevant": is_relevant})

        return {
            "relevance_score": float(top_score),
            "is_relevant": is_relevant
        }

    def query_rewrite_node(self, state: ApexRAGState) -> Dict[str, Any]:
        """Node 4: Self-Correction — rewrite query using key term extraction."""
        original_query = state.get("original_query", state["query"])
        current_query = state["query"]
        loop_count = state["loop_count"] + 1

        span = None
        if self.tracer:
            span = self.tracer.start_span("Query_Rewrite_Expansion", {"loop_count": loop_count})

        # Extract meaningful keywords from original and expand
        stopwords = {'a','an','the','is','in','on','to','for','of','and','or','what','how','which'}
        tokens = [t for t in _tokenize(original_query) if t not in stopwords and len(t) > 2]
        
        # Add domain-specific expansions on each loop
        expansions = ["detailed explanation", "technical overview", "key components"]
        suffix = expansions[loop_count % len(expansions)]
        
        if tokens:
            core = " ".join(tokens[:5])
            rewritten_query = f"{core} {suffix}"
        else:
            rewritten_query = f"{current_query} {suffix}"

        if self.tracer and span:
            self.tracer.end_span(span, {"rewritten_query": rewritten_query})

        return {
            "query": rewritten_query,
            "loop_count": loop_count
        }

    def generate_node(self, state: ApexRAGState) -> Dict[str, Any]:
        """Node 5: Precise Citation-Enforced Answer Generation."""
        query = state["original_query"]
        docs = state["reranked_docs"]

        span = None
        if self.tracer:
            span = self.tracer.start_span("LLM_Generation_With_Citations")

        if not docs:
            response_text = (
                "No relevant context found in the knowledge base for your query.\n\n"
                "Tip: Upload a PDF or document first using the 'Drag & Drop' zone, "
                "then execute your query again."
            )
            citations = []
            prompt_tokens = 20
            completion_tokens = 30
        else:
            # Filter to only chunks with rerank_score above minimum threshold
            MIN_SCORE = 0.0  # keep all, but sort by score descending
            scored_docs = sorted(docs, key=lambda d: d.get("rerank_score", 0.0), reverse=True)

            context_blocks = []
            source_labels = []
            citations = []

            for idx, doc in enumerate(scored_docs, start=1):
                chunk = doc["chunk"]
                chunk_id = chunk["id"]
                source_name = chunk["metadata"].get("source", f"Doc-{idx}")
                page = chunk["metadata"].get("page", None)
                chunk_type = chunk["metadata"].get("chunk_type", "text")
                full_text = chunk["text"]
                score = doc.get("rerank_score", 0.0)

                label = f"Source {idx}: {source_name}"
                if page:
                    label += f" (Page {page})"

                context_blocks.append(full_text)
                source_labels.append(label)

                # Build citation entry with best extracted snippet
                best_snippet = _extract_best_sentences(query, full_text, n=2)
                citations.append({
                    "source_id": f"Source {idx}",
                    "chunk_id": chunk_id,
                    "filename": source_name,
                    "page": page,
                    "chunk_type": chunk_type,
                    "snippet": best_snippet[:200] + ("..." if len(best_snippet) > 200 else ""),
                    "score": score
                })

            # Build the final prompt
            context_str = "\n\n".join(
                f"[Source {i+1}] (File: {source_labels[i]}, Score: {scored_docs[i].get('rerank_score',0):.3f}):\n{block}"
                for i, block in enumerate(context_blocks)
            )
            prompt = (
                f"System: You are a precise document QA assistant. "
                f"Answer ONLY from the provided sources. "
                f"Cite every claim with [Source N]. "
                f"If the sources do not contain the answer, say so explicitly.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Answer (cite sources inline):"
            )

            response_text, prompt_tokens, completion_tokens = self._call_llm(
                prompt, query, context_blocks, source_labels
            )

        if self.tracer and span:
            self.tracer.end_span(span, {"completion_length": len(response_text)})

        return {
            "generation": response_text,
            "citations": citations,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    def _call_llm(
        self,
        prompt: str,
        query: str,
        context_blocks: List[str],
        source_labels: List[str]
    ) -> Tuple[str, int, int]:
        """Call OpenAI LLM if API key set, else use precise extractive synthesis."""

        # ── OpenAI path ──────────────────────────────────────────────────────
        if settings.OPENAI_API_KEY and settings.LLM_PROVIDER == "openai":
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=settings.LLM_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY,
                    temperature=0.0  # deterministic for factual QA
                )
                res = llm.invoke(prompt)
                content = res.content
                token_usage = getattr(res, "response_metadata", {}).get("token_usage", {})
                p_tokens = token_usage.get("prompt_tokens", len(prompt) // 4)
                c_tokens = token_usage.get("completion_tokens", len(content) // 4)
                return content, p_tokens, c_tokens
            except Exception:
                pass  # fall through to extractive synthesis

        # ── Precise Extractive Synthesis (no LLM required) ───────────────────
        answer = _build_direct_answer(query, context_blocks, source_labels)
        p_tokens = len(prompt) // 4
        c_tokens = len(answer) // 4
        return answer, p_tokens, c_tokens
