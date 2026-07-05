import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from graph.state import Young60State
from core.intent_engine import detect_intent
# from core.router import route_query
from core.intent_engine import detect_intent
from core.service_logger import get_service_logger


# ==================================================
# Environment Setup
# ==================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==================================================
# Logger Setup
# ==================================================

logger = get_service_logger("young60_graph")


# ==================================================
# Helpers
# ==================================================

def clean_text(value: str) -> str:
    """
    Clean simple text.
    """

    if not value:
        return ""

    cleaned = str(value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned


def safe_json_loads(raw_text: str) -> dict:
    """
    Safely parse JSON from LLM response.
    """

    if not raw_text:
        return {}

    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    return {}


def is_likely_follow_up(query: str) -> bool:
    """
    Detect if query depends on earlier conversation.
    """

    q = query.lower().strip()

    follow_up_words = [
        "tomorrow",
        "next day",
        "same",
        "there",
        "nearby",
        "near me",
        "what about",
        "and",
        "also",
        "then",
        "that",
        "this",
        "it",
        "uske baad",
        "kal",
        "waha",
        "same place",
        "same city",
        "same medicine",
        "any pharmacy",
        "any hospital",
    ]

    if len(q.split()) <= 4:
        return True

    return any(word in q for word in follow_up_words)


# ==================================================
# Memory Context Node
# ==================================================

def build_memory_context_node(state: Young60State) -> Young60State:
    """
    Build short session memory from recent chat history.
    """

    logger.info("=" * 60)
    logger.info("Young60 graph started")
    logger.info("=" * 60)

    messages = state.get("messages", [])
    session_summary = state.get("session_summary", "")

    recent_messages = messages[-8:]

    memory_lines = []

    for message in recent_messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if not role or not content:
            continue

        content = clean_text(content)

        if len(content) > 700:
            content = content[:700] + "..."

        memory_lines.append(f"{role}: {content}")

    # memory_context = "\n".join(memory_lines)
    
    recent_chat_context = "\n".join(memory_lines)

    if session_summary:
        memory_context = (
            f"Session summary:\n{session_summary}\n\n"
            f"Recent messages:\n{recent_chat_context}"
        )
    else:
        memory_context = recent_chat_context

    logger.info(f"Memory context built with {len(recent_messages)} messages")

    return {
        "memory_context": memory_context
    }


# ==================================================
# Query Rewrite Node
# ==================================================

def rewrite_query_node(state: Young60State) -> Young60State:
    """
    Rewrite follow-up query into a standalone query.

    Example:
    Previous: Weather in Jaipur
    Current: What about tomorrow?
    Rewritten: Weather in Jaipur tomorrow
    """

    user_query = clean_text(state.get("user_query", ""))
    memory_context = state.get("memory_context", "")

    logger.info("Query rewrite node started")
    logger.info(f"Original user query: {user_query}")

    if not user_query:
        return {
            "rewritten_query": "",
            "rewrite_applied": False,
            "rewrite_reason": "Empty query"
        }

    if not memory_context:
        return {
            "rewritten_query": user_query,
            "rewrite_applied": False,
            "rewrite_reason": "No memory context available"
        }

    if not is_likely_follow_up(user_query):
        logger.info("Query appears standalone. Rewrite skipped.")

        return {
            "rewritten_query": user_query,
            "rewrite_applied": False,
            "rewrite_reason": "Standalone query"
        }

    if client is None:
        logger.warning("Query rewrite skipped because OPENAI_API_KEY missing")

        return {
            "rewritten_query": user_query,
            "rewrite_applied": False,
            "rewrite_reason": "OPENAI_API_KEY missing"
        }

    prompt = f"""
You are a query rewriting component for Young60, an assistant for senior citizens.

Your job:
Rewrite the latest user query into a complete standalone query using recent conversation memory.

Return JSON only.

Rules:
- Do not answer the user.
- Do not add new facts.
- Only use information from recent conversation memory.
- If the latest query is already complete, keep it unchanged.
- Preserve the user's language style.
- Preserve intent: weather, medicine, nearby places, metro/public transport, or general help.
- For weather follow-ups, carry forward the city/place if clear.
- For nearby follow-ups, carry forward the last location if clear.
- For medicine follow-ups, carry forward the medicine name only if clear.
- For metro follow-ups, carry forward source/destination only if clear.
- If unclear, keep the original query unchanged.
- Do not store or expose sensitive information.

Return JSON in this exact shape:
{{"rewritten_query":"...", "rewrite_applied":true, "rewrite_reason":"..."}}

Examples:

Memory:
user: Weather in Jaipur
assistant: Jaipur weather is hot today.
Latest user query:
What about tomorrow?
JSON:
{{"rewritten_query":"Weather in Jaipur tomorrow","rewrite_applied":true,"rewrite_reason":"Carried forward city Jaipur from previous weather query"}}

Memory:
user: Find hospital near Rohini
assistant: Here are hospitals near Rohini.
Latest user query:
Any pharmacy nearby?
JSON:
{{"rewritten_query":"Find pharmacy near Rohini","rewrite_applied":true,"rewrite_reason":"Carried forward location Rohini from previous nearby query"}}

Memory:
user: Dolo 650 use
assistant: Dolo 650 is generally used for fever/pain awareness.
Latest user query:
side effects?
JSON:
{{"rewritten_query":"Dolo 650 side effects","rewrite_applied":true,"rewrite_reason":"Carried forward medicine name Dolo 650"}}

Memory:
user: How to use WhatsApp video call?
assistant: Open WhatsApp and tap video icon.
Latest user query:
Explain slowly
JSON:
{{"rewritten_query":"Explain how to use WhatsApp video call slowly","rewrite_applied":true,"rewrite_reason":"Carried forward previous digital help topic"}}

Recent conversation memory:
{memory_context}

Latest user query:
{user_query}
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        logger.info(f"Raw rewrite response: {raw_text}")

        parsed = safe_json_loads(raw_text)

        rewritten_query = clean_text(parsed.get("rewritten_query", user_query))
        rewrite_applied = bool(parsed.get("rewrite_applied", False))
        rewrite_reason = clean_text(parsed.get("rewrite_reason", ""))

        if not rewritten_query:
            rewritten_query = user_query
            rewrite_applied = False
            rewrite_reason = "Rewrite returned empty query"

        logger.info(f"Rewritten query: {rewritten_query}")
        logger.info(f"Rewrite applied: {rewrite_applied}")
        logger.info(f"Rewrite reason: {rewrite_reason}")

        return {
            "rewritten_query": rewritten_query,
            "rewrite_applied": rewrite_applied,
            "rewrite_reason": rewrite_reason
        }

    except Exception as error:
        logger.error(f"Query rewrite failed: {error}")

        return {
            "rewritten_query": user_query,
            "rewrite_applied": False,
            "rewrite_reason": f"Rewrite failed: {error}"
        }

# ==================================================
# Safety Guard Node
# ==================================================

def safety_guard_node(state: Young60State) -> Young60State:
    """
    Check urgent or sensitive queries before service routing.

    This is especially important for senior citizen use cases.
    """

    query = state.get("rewritten_query") or state.get("user_query", "")
    q = query.lower().strip()

    logger.info("Safety guard node started")
    logger.info(f"Safety query: {query}")

    emergency_health_words = [
        "chest pain",
        "breathing problem",
        "difficulty breathing",
        "can't breathe",
        "cannot breathe",
        "unconscious",
        "fainted",
        "stroke",
        "heart attack",
        "severe bleeding",
        "accident",
        "poison",
        "overdose",
        "too much medicine",
        "medicine overdose",
        "suicide",
        "self harm",
        "marna",
        "saans nahi",
        "saans ni",
        "behosh",
    ]

    banking_sensitive_words = [
        "share otp",
        "give otp",
        "otp share",
        "upi pin",
        "atm pin",
        "card pin",
        "cvv",
        "password share",
        "screen sharing",
        "anydesk",
        "teamviewer",
    ]

    if any(word in q for word in emergency_health_words):
        logger.warning("Emergency health/safety query detected")

        return {
            "safety_blocked": True,
            "safety_reason": "Emergency health/safety query",
            "service_node": "safety_guard",
            "raw_response": (
                "This may be urgent.\n\n"
                "Please contact local emergency services or a nearby doctor/hospital immediately.\n\n"
                "Do not wait only for an app response.\n\n"
                "If someone has chest pain, breathing difficulty, is unconscious, has severe bleeding, "
                "or may have taken too much medicine, get medical help now."
            )
        }

    if any(word in q for word in banking_sensitive_words):
        logger.warning("Banking/OTP safety query detected")

        return {
            "safety_blocked": True,
            "safety_reason": "Banking/OTP sensitive query",
            "service_node": "safety_guard",
            "raw_response": (
                "Please do not share OTP, UPI PIN, ATM PIN, CVV, password, or screen-sharing access with anyone.\n\n"
                "A real bank or government officer will not ask for these details.\n\n"
                "If money is involved or you already shared details, contact your bank immediately and block the transaction/card if needed."
            )
        }

    return {
        "safety_blocked": False,
        "safety_reason": ""
    }


def route_after_safety(state: Young60State) -> str:
    """
    Decide whether to continue routing or stop at safety response.
    """

    if state.get("safety_blocked"):
        logger.info("Safety guard blocked normal routing")
        return "final_response"

    logger.info("Safety guard passed. Continuing to intent detection")
    return "detect_intent"

# ==================================================
# Intent Detection Node
# ==================================================

def detect_intent_node(state: Young60State) -> Young60State:
    """
    Detect intent using rewritten query.
    """

    query = state.get("rewritten_query") or state.get("user_query", "")

    logger.info("Intent detection node started")
    logger.info(f"Intent query: {query}")

    intent = detect_intent(query)

    logger.info(f"Graph detected intent: {intent}")

    return {
        "intent": intent
    }

# ==================================================
# Conditional Routing
# ==================================================

def route_by_intent(state: Young60State) -> str:
    """
    Decide which graph node should handle the query.
    """

    intent = state.get("intent", "general")

    logger.info(f"Conditional route selected for intent: {intent}")

    if intent == "weather":
        return "weather_node"

    if intent == "medicine":
        return "medicine_node"

    if intent == "nearby":
        return "nearby_node"

    if intent == "metro":
        return "metro_node"

    return "general_node"



# ==================================================
# Service Nodes
# ==================================================

def weather_node(state: Young60State) -> Young60State:
    """
    Weather service node.
    """

    query = state.get("rewritten_query") or state.get("user_query", "")

    logger.info("Weather node started")
    logger.info(f"Weather query: {query}")

    try:
        from services.weather_service import weather_help

        response = weather_help(query)

        return {
            "raw_response": response,
            "service_node": "weather_node"
        }

    except Exception as error:
        logger.error(f"Weather node failed: {error}")

        return {
            "raw_response": (
                "Sorry, I could not fetch weather information right now.\n\n"
                f"Technical detail: `{error}`"
            ),
            "service_node": "weather_node",
            "error": str(error)
        }


def medicine_node(state: Young60State) -> Young60State:
    """
    Medicine service node.
    """

    query = state.get("rewritten_query") or state.get("user_query", "")

    logger.info("Medicine node started")
    logger.info(f"Medicine query: {query}")

    try:
        from services.medicine_service import medicine_help

        response = medicine_help(query)

        return {
            "raw_response": response,
            "service_node": "medicine_node"
        }

    except Exception as error:
        logger.error(f"Medicine node failed: {error}")

        return {
            "raw_response": (
                "Sorry, I could not prepare medicine information right now.\n\n"
                f"Technical detail: `{error}`"
            ),
            "service_node": "medicine_node",
            "error": str(error)
        }


def nearby_node(state: Young60State) -> Young60State:
    """
    Nearby places service node.
    """

    query = state.get("rewritten_query") or state.get("user_query", "")
    user_location = state.get("user_location")

    logger.info("Nearby node started")
    logger.info(f"Nearby query: {query}")

    try:
        from services.nearby_service import nearby_help

        response = nearby_help(
            query=query,
            user_location=user_location
        )

        return {
            "raw_response": response,
            "service_node": "nearby_node"
        }

    except Exception as error:
        logger.error(f"Nearby node failed: {error}")

        return {
            "raw_response": (
                "Sorry, I could not find nearby places right now.\n\n"
                f"Technical detail: `{error}`"
            ),
            "service_node": "nearby_node",
            "error": str(error)
        }


def metro_node(state: Young60State) -> Young60State:
    """
    Metro/public transport service node.
    """

    query = state.get("rewritten_query") or state.get("user_query", "")

    logger.info("Metro node started")
    logger.info(f"Metro query: {query}")

    try:
        from services.metro_service import metro_help

        response = metro_help(query)

        return {
            "raw_response": response,
            "service_node": "metro_node"
        }

    except Exception as error:
        logger.error(f"Metro node failed: {error}")

        return {
            "raw_response": (
                "Sorry, I could not prepare metro/public transport route right now.\n\n"
                f"Technical detail: `{error}`"
            ),
            "service_node": "metro_node",
            "error": str(error)
        }


def general_node(state: Young60State) -> Young60State:
    """
    General service node.
    """

    query = state.get("rewritten_query") or state.get("user_query", "")

    logger.info("General node started")
    logger.info(f"General query: {query}")

    try:
        from services.general_service import general_help

        response = general_help(query)

        return {
            "raw_response": response,
            "service_node": "general_node"
        }

    except Exception as error:
        logger.error(f"General node failed: {error}")

        return {
            "raw_response": (
                "Sorry, I could not prepare a general answer right now.\n\n"
                f"Technical detail: `{error}`"
            ),
            "service_node": "general_node",
            "error": str(error)
        }

# ==================================================
# Final Response Node
# ==================================================

def clean_final_response(text: str) -> str:
    """
    Remove internal/debug/source lines from final user-facing response.
    """

    if not text:
        return ""

    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip().lower()

        if stripped.startswith("internal"):
            continue

        if stripped.startswith("source:"):
            continue

        if stripped.startswith("api path"):
            continue

        if stripped.startswith("debug"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def final_response_node(state: Young60State) -> Young60State:
    """
    Final guard node before response is shown to user.

    Responsibilities:
    1. Clean final response
    2. Handle blank response
    3. Build graph trace for debugging
    """

    logger.info("Final response node started")

    raw_response = state.get("raw_response", "")
    final_response = clean_final_response(raw_response)

    if not final_response:
        final_response = (
            "Sorry, I could not prepare a proper answer for this.\n\n"
            "Please try again with a little more detail."
        )

    graph_trace = {
        "user_query": state.get("user_query", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "rewrite_applied": state.get("rewrite_applied", False),
        "rewrite_reason": state.get("rewrite_reason", ""),
        "safety_blocked": state.get("safety_blocked", False),
        "safety_reason": state.get("safety_reason", ""),
        "intent": state.get("intent", ""),
        "service_node": state.get("service_node", ""),
        "has_user_location": bool(state.get("user_location")),
        "error": state.get("error", "")
    }

    logger.info(f"Graph trace: {graph_trace}")

    return {
        "response": final_response,
        "graph_trace": graph_trace
    }

# ==================================================
# Session Summary Node
# ==================================================

def update_session_summary_node(state: Young60State) -> Young60State:
    """
    Update short session summary after each assistant response.

    This is session-only memory.
    It is not saved permanently.
    """

    logger.info("Session summary node started")

    previous_summary = state.get("session_summary", "")
    user_query = state.get("user_query", "")
    rewritten_query = state.get("rewritten_query", "")
    intent = state.get("intent", "")
    response = state.get("response", "")

    if client is None:
        logger.warning("Session summary skipped because OPENAI_API_KEY missing")
        return {
            "updated_session_summary": previous_summary
        }

    prompt = f"""
You are maintaining a short session memory summary for Young60.

Rules:
- Keep only useful context for follow-up questions.
- Keep it short, maximum 120 words.
- Include latest topic, location, medicine name, or user preference only if useful.
- Do not store OTP, PIN, password, card number, address, or sensitive personal details.
- Do not store medical diagnosis.
- Do not add facts not present in conversation.
- This is only temporary session memory.

Previous session summary:
{previous_summary}

Latest user query:
{user_query}

Rewritten query:
{rewritten_query}

Detected intent:
{intent}

Assistant response:
{response}

Return only the updated summary text.
"""

    try:
        llm_response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        updated_summary = llm_response.choices[0].message.content.strip()

        logger.info(f"Updated session summary: {updated_summary}")

        return {
            "updated_session_summary": updated_summary
        }

    except Exception as error:
        logger.error(f"Session summary update failed: {error}")

        return {
            "updated_session_summary": previous_summary
        }



# ==================================================
# Build Graph
# ==================================================

def build_young60_graph():
    """
    Build and compile Young60 LangGraph.
    """

    graph_builder = StateGraph(Young60State)

    graph_builder.add_node("build_memory_context", build_memory_context_node)
    graph_builder.add_node("rewrite_query", rewrite_query_node)
    graph_builder.add_node("detect_intent", detect_intent_node)

    graph_builder.add_node("weather_node", weather_node)
    graph_builder.add_node("medicine_node", medicine_node)
    graph_builder.add_node("nearby_node", nearby_node)
    graph_builder.add_node("metro_node", metro_node)
    graph_builder.add_node("general_node", general_node)

    graph_builder.add_node("final_response", final_response_node)
    graph_builder.add_node("update_session_summary", update_session_summary_node)
    graph_builder.add_node("safety_guard", safety_guard_node)

    graph_builder.add_edge(START, "build_memory_context")
    graph_builder.add_edge("build_memory_context", "rewrite_query")
    graph_builder.add_edge("rewrite_query", "safety_guard")

    graph_builder.add_conditional_edges(
        "safety_guard",
        route_after_safety,
        {
            "detect_intent": "detect_intent",
            "final_response": "final_response"
        }
    )

    graph_builder.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {
            "weather_node": "weather_node",
            "medicine_node": "medicine_node",
            "nearby_node": "nearby_node",
            "metro_node": "metro_node",
            "general_node": "general_node"
        }
    )

    graph_builder.add_edge("weather_node", "final_response")
    graph_builder.add_edge("medicine_node", "final_response")
    graph_builder.add_edge("nearby_node", "final_response")
    graph_builder.add_edge("metro_node", "final_response")
    graph_builder.add_edge("general_node", "final_response")

    graph_builder.add_edge("final_response", "update_session_summary")
    graph_builder.add_edge("update_session_summary", END)

    return graph_builder.compile()

young60_graph = build_young60_graph()