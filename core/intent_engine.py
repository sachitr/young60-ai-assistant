from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from core.intent_keyword_store import load_intent_keywords


# ==================================================
# Setup
# ==================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
MEDICINE_ALIAS_PATH = BASE_DIR / "data" / "medicine_aliases.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==================================================
# Helpers
# ==================================================

def normalize_query(query: str) -> str:
    if not query:
        return ""

    return query.lower().strip()


def load_medicine_aliases() -> list:
    """
    Load known medicine brand names from medicine_aliases.csv.
    This helps detect medicine intent quickly without LLM.
    """

    try:
        df = pd.read_csv(MEDICINE_ALIAS_PATH, encoding="utf-8-sig")
        df.columns = [col.strip().lower() for col in df.columns]

        if "brand_name" not in df.columns:
            return []

        return (
            df["brand_name"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .tolist()
        )

    except Exception:
        return []


# ==================================================
# Fast Rule-Based Detection
# ==================================================
def detect_intent_by_rules(query: str) -> str:
    """
    Fast intent detection using priority rules + CSV keywords.

    Important:
    Nearby search should win over metro when query contains:
    hospital/pharmacy/atm/library/etc. near a station/place.
    """

    q = normalize_query(query)

    # --------------------------------------------------
    # Strong nearby detection first
    # --------------------------------------------------

    nearby_place_words = [
        "hospital",
        "pharmacy",
        "chemist",
        "medical store",
        "clinic",
        "doctor",
        "atm",
        "bank",
        "police",
        "police station",
        "grocery",
        "supermarket",
        "restaurant",
        "park",
        "library",
        "nearby",
        "near me",
        "around me",
    ]

    nearby_action_words = [
        "find",
        "show",
        "near",
        "nearby",
        "around",
        "closest",
        "nearest",
        "within",
    ]

    if any(place in q for place in nearby_place_words) and any(action in q for action in nearby_action_words):
        print("[INTENT_ENGINE] Strong nearby intent detected", flush=True)
        return "nearby"

    # --------------------------------------------------
    # Medicine detection
    # --------------------------------------------------

    medicine_words = [
        "medicine",
        "tablet",
        "capsule",
        "syrup",
        "dose",
        "dosage",
        "side effect",
        "side effects",
        "used for",
        "usage",
        "use of",
        "can i take",
        "interaction",
        "composition",
        "ingredient",
        "dawai",
        "dawa",
    ]

    if any(word in q for word in medicine_words):
        print("[INTENT_ENGINE] Strong medicine intent detected", flush=True)
        return "medicine"

    # --------------------------------------------------
    # Weather detection
    # --------------------------------------------------

    weather_words = [
        "weather",
        "temperature",
        "rain",
        "snow",
        "storm",
        "thunderstorm",
        "hot",
        "heat",
        "forecast",
        "mausam",
    ]

    if any(word in q for word in weather_words):
        print("[INTENT_ENGINE] Strong weather intent detected", flush=True)
        return "weather"

    # --------------------------------------------------
    # Metro detection
    # --------------------------------------------------

    metro_words = [
        "metro",
        "delhi metro",
        "metro route",
        "metro line",
        "interchange",
        "yellow line",
        "blue line",
        "pink line",
        "magenta line",
    ]

    if any(word in q for word in metro_words):
        print("[INTENT_ENGINE] Strong metro intent detected", flush=True)
        return "metro"

    # --------------------------------------------------
    # CSV keyword fallback
    # --------------------------------------------------

    keyword_map = load_intent_keywords()

    matched_intents = []

    for intent, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword and keyword in q:
                matched_intents.append(intent)
                print(
                    f"[INTENT_ENGINE] CSV keyword match: '{keyword}' → {intent}",
                    flush=True
                )

    if matched_intents:
        priority = ["nearby", "medicine", "weather", "metro", "general"]

        for intent in priority:
            if intent in matched_intents:
                print(f"[INTENT_ENGINE] Final priority intent: {intent}", flush=True)
                return intent

    return ""

# ==================================================
# LLM Fallback Detection
# ==================================================

def detect_intent_with_llm(query: str) -> str:
    """
    LLM fallback intent detection.
    Used only when rule-based detection fails.
    """

    if client is None:
        return "general"

    prompt = f"""
You are the intent router for Young60, an AI assistant for senior citizens in India.

Classify the user query into exactly one category:

metro
medicine
nearby
weather
general

Definitions:
- metro: route, station, Delhi Metro, travel between metro stations
- medicine: medicine use, tablets, side effects, dosage question, pain balm, fever medicine
- nearby: find hospital, pharmacy, clinic, ATM, bank, doctor near user
- weather: weather, rain, temperature, mausam
- general: anything else

Examples:
Query: How do I go from Rohini to AIIMS?
Answer: metro

Query: Usage for Volini
Answer: medicine

Query: Combiflam ka kya use hai?
Answer: medicine

Query: Find hospital near me
Answer: nearby

Query: Aaj mausam kaisa hai?
Answer: weather

Query: How to use WhatsApp?
Answer: general

User query:
{query}

Return only one word:
metro, medicine, nearby, weather, or general.
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

        intent = response.choices[0].message.content.strip().lower()

        allowed_intents = ["metro", "medicine", "nearby", "weather", "general"]

        if intent in allowed_intents:
            return intent

        return "general"

    except Exception:
        return "general"


# ==================================================
# Main Intent Function
# ==================================================

def detect_intent(query: str) -> str:
    """
    Main intent detection function.

    Flow:
    1. Fast keyword/entity rules
    2. LLM fallback if unclear
    3. Default to general
    """

    rule_intent = detect_intent_by_rules(query)

    if rule_intent:
        print(f"[INTENT_ENGINE] Rule intent detected: {rule_intent}", flush=True)
        return rule_intent

    print("[INTENT_ENGINE] No rule match. Calling LLM intent fallback.", flush=True)

    llm_intent = detect_intent_with_llm(query)

    print(f"[INTENT_ENGINE] LLM intent detected: {llm_intent}", flush=True)

    return llm_intent