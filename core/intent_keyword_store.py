from pathlib import Path
from datetime import datetime
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
KEYWORD_FILE = DATA_DIR / "intent_keywords.csv"


VALID_INTENTS = {"metro", "medicine", "nearby", "weather", "general"}


def load_intent_keywords() -> dict:
    """
    Load intent keywords from data/intent_keywords.csv.

    Returns:
        {
            "metro": ["metro", "route"],
            "medicine": ["tablet", "dose"]
        }
    """

    if not KEYWORD_FILE.exists():
        return {}

    try:
        df = pd.read_csv(KEYWORD_FILE, encoding="utf-8-sig")
        df.columns = [col.strip().lower() for col in df.columns]

        required_columns = ["intent", "keyword", "active"]

        for col in required_columns:
            if col not in df.columns:
                return {}

        df = df.fillna("")
        df = df[df["active"].astype(str).str.lower() == "yes"]

        keyword_map = {}

        for _, row in df.iterrows():
            intent = str(row["intent"]).strip().lower()
            keyword = str(row["keyword"]).strip().lower()

            if not intent or not keyword:
                continue

            if intent not in VALID_INTENTS:
                continue

            keyword_map.setdefault(intent, []).append(keyword)

        return keyword_map

    except Exception as error:
        print(f"[INTENT_KEYWORD_STORE] Error loading keywords: {error}", flush=True)
        return {}


def add_intent_keyword(intent: str, keyword: str, notes: str = "") -> bool:
    """
    Add a new keyword to data/intent_keywords.csv.

    This is admin-style update.
    Do not call this automatically for every user query.
    """

    intent = intent.strip().lower()
    keyword = keyword.strip().lower()

    if intent not in VALID_INTENTS:
        print(f"[INTENT_KEYWORD_STORE] Invalid intent: {intent}", flush=True)
        return False

    if not keyword:
        print("[INTENT_KEYWORD_STORE] Empty keyword not allowed", flush=True)
        return False

    DATA_DIR.mkdir(exist_ok=True)

    if KEYWORD_FILE.exists():
        df = pd.read_csv(KEYWORD_FILE, encoding="utf-8-sig")
        df.columns = [col.strip().lower() for col in df.columns]
    else:
        df = pd.DataFrame(columns=["intent", "keyword", "active", "notes", "created_at"])

    df = df.fillna("")

    existing = df[
        (df["intent"].astype(str).str.lower() == intent) &
        (df["keyword"].astype(str).str.lower() == keyword)
    ]

    if not existing.empty:
        print("[INTENT_KEYWORD_STORE] Keyword already exists", flush=True)
        return False

    new_row = {
        "intent": intent,
        "keyword": keyword,
        "active": "yes",
        "notes": notes,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(KEYWORD_FILE, index=False, encoding="utf-8-sig")

    print(
        f"[INTENT_KEYWORD_STORE] Added keyword '{keyword}' for intent '{intent}'",
        flush=True
    )

    return True