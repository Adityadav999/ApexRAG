from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from apexrag.config import settings
from apexrag.graph.state import ApexRAGState
from apexrag.graph.nodes import GraphNodes
from apexrag.retrieval.hybrid import HybridRetriever
from apexrag.reranking.cross_encoder import CrossEncoderReranker
from apexrag.observability.tracer import ApexTracer

def route_after_grading(state: ApexRAGState) -> str:
    """Conditional Edge decision function after grading context relevance."""
    if state.get("is_relevant", False) or state.get("loop_count", 0) >= settings.MAX_SELF_CORRECTION_LOOPS:
        return "generate"
    return "rewrite"

def build_apexrag_graph(
    hybrid_retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    tracer: Optional[ApexTracer] = None
):
    """Construct and compile the ApexRAG stateful LangGraph workflow."""
    nodes = GraphNodes(hybrid_retriever, reranker, tracer)
    
    workflow = StateGraph(ApexRAGState)

    # Define Graph Nodes
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("rerank", nodes.rerank_node)
    workflow.add_node("grade_relevance", nodes.grade_relevance_node)
    workflow.add_node("query_rewrite", nodes.query_rewrite_node)
    workflow.add_node("generate", nodes.generate_node)

    # Set Entry Point
    workflow.set_entry_point("retrieve")

    # Connect Edges
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_relevance")

    # Add Conditional Edge from grade_relevance
    workflow.add_conditional_edges(
        "grade_relevance",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "query_rewrite"
        }
    )

    # Re-route query_rewrite back to retrieve
    workflow.add_edge("query_rewrite", "retrieve")

    # End workflow after generation
    workflow.add_edge("generate", END)

    return workflow.compile()
