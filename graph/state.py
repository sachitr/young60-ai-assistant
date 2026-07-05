from typing import TypedDict, List, Dict, Optional, Any


class Young60State(TypedDict, total=False):
    """
    Shared state for Young60 LangGraph.

    This state travels through all graph nodes.
    """

    user_query: str
    rewritten_query: str
    rewrite_applied: bool
    rewrite_reason: str

    messages: List[Dict[str, str]]
    memory_context: str

    session_summary: str
    updated_session_summary: str

    user_location: Optional[Dict[str, Any]]

    safety_blocked: bool
    safety_reason: str

    intent: str
    service_node: str

    raw_response: str
    response: str

    graph_trace: Dict[str, Any]

    error: str