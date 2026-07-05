from pathlib import Path
from difflib import get_close_matches
import os
import json
import re

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

from core.service_logger import get_service_logger


# ==================================================
# Environment Setup
# ==================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

ALIAS_PATH = DATA_DIR / "medicine_aliases.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==================================================
# Logger Setup
# ==================================================

logger = get_service_logger("medicine_service")


# ==================================================
# Basic Helpers
# ==================================================

def clean_text(value: str) -> str:
    """
    Clean basic text values.
    """

    if not value:
        return ""

    cleaned = str(value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned


def safe_json_loads(raw_text: str) -> dict:
    """
    Parse JSON safely from LLM response.
    Handles plain JSON and accidental markdown code blocks.
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


def simplify_text(text: str, max_sentences: int = 2) -> str:
    """
    Keep medical label text short and readable.
    This is a formatter, not medical reasoning.
    """

    if not text:
        return "Information not available from the connected source."

    cleaned = text.replace("\n", " ").replace("  ", " ").strip()

    sentences = cleaned.split(".")
    selected = []

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence:
            selected.append(sentence)

        if len(selected) >= max_sentences:
            break

    if not selected:
        return cleaned[:300]

    return ". ".join(selected) + "."


# ==================================================
# Medicine Query Parser
# ==================================================

def parse_medicine_query_with_llm(query: str) -> dict:
    """
    Use LLM to parse the user's medicine query into structured JSON.

    Important:
    LLM is only used for understanding the query.
    Medical facts still come from CSV/openFDA.
    """

    logger.info("LLM medicine query parser started")
    logger.info(f"Parser input query: {query}")

    default_result = {
        "medicine_name": "",
        "second_medicine_name": "",
        "query_type": "general",
        "response_format": "text"
    }

    if client is None:
        logger.warning("LLM medicine parser skipped because OPENAI_API_KEY is missing")
        return default_result

    prompt = f"""
You are a medicine query parser for Young60, an assistant for senior citizens.

Convert the user's medicine question into JSON only.

Allowed query_type values:
- general
- usage
- side_effects
- warnings
- dosage
- interactions
- ingredients
- pregnancy
- overdose
- storage

Allowed response_format values:
- text
- table

Rules:
- Extract only the medicine name or brand name.
- If user compares or combines two medicines, put the second one in second_medicine_name.
- If no medicine is mentioned, medicine_name should be empty string.
- If user asks "used for", "ka use", "kis kaam", use query_type usage.
- If user asks side effects, use side_effects.
- If user asks warnings, precautions, safe or unsafe, use warnings.
- If user asks dose, dosage, how much, when to take, use dosage.
- If user asks can I take X with Y, use interactions.
- If user asks ingredient/composition, use ingredients.
- If user asks pregnancy/breastfeeding, use pregnancy.
- If user asks overdose, use overdose.
- If user asks storage, use storage.
- If user asks table, day-wise, comparison table, use response_format table.
- If user asks bullets/list/points/WhatsApp note, use response_format text.
- Return valid JSON only. No markdown. No explanation.

Examples:

User: What is Dolo 650 used for?
JSON:
{{"medicine_name":"Dolo 650","second_medicine_name":"","query_type":"usage","response_format":"text"}}

User: Crocin side effects in table
JSON:
{{"medicine_name":"Crocin","second_medicine_name":"","query_type":"side_effects","response_format":"table"}}

User: Can I take Combiflam with Dolo?
JSON:
{{"medicine_name":"Combiflam","second_medicine_name":"Dolo","query_type":"interactions","response_format":"text"}}

User: Volini ka use kya hai, WhatsApp note banao
JSON:
{{"medicine_name":"Volini","second_medicine_name":"","query_type":"usage","response_format":"text"}}

User: Dolo dosage?
JSON:
{{"medicine_name":"Dolo","second_medicine_name":"","query_type":"dosage","response_format":"text"}}

User query:
{query}
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

        logger.info(f"Raw medicine parser response: {raw_text}")

        parsed = safe_json_loads(raw_text)

        medicine_name = clean_text(parsed.get("medicine_name", ""))
        second_medicine_name = clean_text(parsed.get("second_medicine_name", ""))

        query_type = str(parsed.get("query_type", "general")).strip().lower()
        response_format = str(parsed.get("response_format", "text")).strip().lower()

        allowed_query_types = {
            "general",
            "usage",
            "side_effects",
            "warnings",
            "dosage",
            "interactions",
            "ingredients",
            "pregnancy",
            "overdose",
            "storage",
        }

        if query_type not in allowed_query_types:
            query_type = "general"

        if response_format not in {"text", "table"}:
            response_format = "text"

        result = {
            "medicine_name": medicine_name,
            "second_medicine_name": second_medicine_name,
            "query_type": query_type,
            "response_format": response_format
        }

        logger.info(f"Parsed medicine query: {result}")

        return result

    except Exception as error:
        logger.error(f"LLM medicine parser failed: {error}")
        return default_result


def parse_medicine_query_by_rules(query: str, alias_df: pd.DataFrame) -> dict:
    """
    Fallback parser when LLM is unavailable or fails.
    """

    logger.info("Rule-based medicine query parser started")

    q = query.lower().strip()

    result = {
        "medicine_name": "",
        "second_medicine_name": "",
        "query_type": "general",
        "response_format": "text"
    }

    if any(word in q for word in ["used for", "usage", "use of", "ka use", "kis kaam", "kis liye"]):
        result["query_type"] = "usage"

    elif any(word in q for word in ["side effect", "side effects", "adverse"]):
        result["query_type"] = "side_effects"

    elif any(word in q for word in ["warning", "warnings", "safe", "unsafe", "precaution", "precautions"]):
        result["query_type"] = "warnings"

    elif any(word in q for word in ["dose", "dosage", "how much", "when to take", "kitni"]):
        result["query_type"] = "dosage"

    elif any(word in q for word in ["with", "together", "combine", "interaction", "interactions"]):
        result["query_type"] = "interactions"

    elif any(word in q for word in ["ingredient", "ingredients", "composition", "contains"]):
        result["query_type"] = "ingredients"

    elif any(word in q for word in ["pregnancy", "pregnant", "breastfeeding"]):
        result["query_type"] = "pregnancy"

    elif "overdose" in q:
        result["query_type"] = "overdose"

    elif "storage" in q or "store" in q:
        result["query_type"] = "storage"

    if any(word in q for word in ["table", "table format", "comparison table"]):
        result["response_format"] = "table"

    # Try alias match from CSV
    medicine = find_medicine_alias(query, alias_df)

    if medicine is not None:
        result["medicine_name"] = str(medicine["brand_name"]).strip()

    logger.info(f"Rule parser result: {result}")

    return result


def parse_medicine_query(query: str, alias_df: pd.DataFrame) -> dict:
    """
    Main medicine query parser.

    Flow:
    1. Try LLM parser
    2. Use rule parser as fallback or enrichment
    """

    llm_result = parse_medicine_query_with_llm(query)
    rule_result = parse_medicine_query_by_rules(query, alias_df)

    final_result = llm_result.copy()

    if not final_result.get("medicine_name") and rule_result.get("medicine_name"):
        final_result["medicine_name"] = rule_result["medicine_name"]

    if final_result.get("query_type") == "general" and rule_result.get("query_type") != "general":
        final_result["query_type"] = rule_result["query_type"]

    if rule_result.get("response_format") == "table":
        final_result["response_format"] = "table"

    logger.info(f"Final parsed medicine query: {final_result}")

    return final_result


# ==================================================
# Load Local Alias File
# ==================================================

def load_alias_data() -> pd.DataFrame:
    """
    Load local medicine alias file.

    Expected file:
    data/medicine_aliases.csv

    Expected columns:
    brand_name,generic_name,common_use,notes
    """

    logger.info("Loading medicine alias file")

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"]

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(ALIAS_PATH, encoding=encoding)
            df.columns = [col.strip().lower() for col in df.columns]

            required_columns = ["brand_name", "generic_name", "common_use", "notes"]

            for col in required_columns:
                if col not in df.columns:
                    df[col] = ""

            df = df.fillna("")

            logger.info(f"Medicine alias file loaded successfully with encoding: {encoding}")
            logger.info(f"Medicine alias records loaded: {len(df)}")

            return df

        except UnicodeDecodeError:
            continue

        except FileNotFoundError:
            logger.error("medicine_aliases.csv not found")
            return pd.DataFrame(columns=["brand_name", "generic_name", "common_use", "notes"])

        except Exception as error:
            logger.error(f"Unexpected error while loading alias file: {error}")
            return pd.DataFrame(columns=["brand_name", "generic_name", "common_use", "notes"])

    logger.error("Could not read medicine_aliases.csv with supported encodings")
    return pd.DataFrame(columns=["brand_name", "generic_name", "common_use", "notes"])


# ==================================================
# Alias Search
# ==================================================

def find_medicine_alias(search_text: str, alias_df: pd.DataFrame):
    """
    Search medicine in local alias CSV.

    Search order:
    1. Brand name exact/partial match
    2. Generic name exact/partial match
    3. Fuzzy match on brand name
    """

    if alias_df.empty or not search_text:
        return None

    search_lower = search_text.lower().strip()

    logger.info(f"Searching medicine in local alias file for: {search_text}")

    for _, row in alias_df.iterrows():
        brand = str(row["brand_name"]).lower().strip()
        generic = str(row["generic_name"]).lower().strip()

        if brand and brand in search_lower:
            logger.info(f"Alias match found by brand name: {row['brand_name']}")
            return row

        if generic and generic in search_lower:
            logger.info(f"Alias match found by generic name: {row['generic_name']}")
            return row

    brand_names = alias_df["brand_name"].dropna().astype(str).tolist()

    close_matches = get_close_matches(
        search_text,
        brand_names,
        n=1,
        cutoff=0.55
    )

    if close_matches:
        matched_brand = close_matches[0]
        logger.info(f"Fuzzy alias match found: {matched_brand}")
        return alias_df[alias_df["brand_name"] == matched_brand].iloc[0]

    logger.info("No local alias match found")
    return None


# ==================================================
# openFDA API Lookup
# ==================================================

def fetch_openfda_label(medicine_name: str):
    """
    Fetch public drug label information from openFDA.

    Note:
    openFDA is not India-specific.
    It is used only for general awareness.
    """

    if not medicine_name:
        logger.warning("openFDA lookup skipped because medicine name is empty")
        return None

    logger.info(f"openFDA lookup started for: {medicine_name}")

    url = "https://api.fda.gov/drug/label.json"

    search_queries = [
        f'openfda.generic_name:"{medicine_name}"',
        f'openfda.brand_name:"{medicine_name}"',
        medicine_name
    ]

    for search_query in search_queries:
        params = {
            "search": search_query,
            "limit": 1
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            logger.info(
                f"openFDA status: {response.status_code} | search: {search_query}"
            )

            if response.status_code != 200:
                continue

            data = response.json()

            if "results" in data and data["results"]:
                logger.info("openFDA result found")
                return data["results"][0]

        except requests.RequestException as error:
            logger.error(f"openFDA request failed: {error}")
            continue

        except Exception as error:
            logger.error(f"Unexpected openFDA error: {error}")
            continue

    logger.warning(f"No openFDA result found for: {medicine_name}")
    return None


# ==================================================
# Label Helpers
# ==================================================

def get_first_section(label: dict, keys: list, max_chars: int = 900) -> str:
    """
    Extract first available section from openFDA label response.
    """

    for key in keys:
        value = label.get(key)

        if isinstance(value, list) and value:
            return str(value[0]).strip()[:max_chars]

        if isinstance(value, str) and value.strip():
            return value.strip()[:max_chars]

    return ""


def get_openfda_names(label: dict) -> dict:
    """
    Extract generic/brand names from openFDA metadata.
    """

    openfda = label.get("openfda", {}) if label else {}

    generic_names = openfda.get("generic_name", [])
    brand_names = openfda.get("brand_name", [])

    generic_name = generic_names[0] if generic_names else ""
    brand_name = brand_names[0] if brand_names else ""

    return {
        "generic_name": generic_name,
        "brand_name": brand_name
    }


# ==================================================
# Fact Builder
# ==================================================
##LLM fallback
def generate_llm_fallback_medicine_info(
    medicine_name: str,
    query_type: str,
    second_medicine_name: str = ""
) -> str:
    """
    Controlled LLM fallback when CSV/openFDA do not have medicine data.

    This is only for general awareness.
    It must not provide personal dosage, interaction approval, or emergency advice.
    """

    if client is None:
        logger.warning("LLM fallback medicine info skipped because OPENAI_API_KEY is missing")
        return ""

    logger.info(f"LLM fallback medicine info started for: {medicine_name}")

    prompt = f"""
You are helping Young60, an assistant for senior citizens.

The connected verified medicine API did not return data for this medicine.

Medicine name:
{medicine_name}

Second medicine name, if any:
{second_medicine_name}

User query type:
{query_type}

Rules:
- Give only general awareness information.
- Do not provide personal dosage.
- Do not say it is safe to combine medicines.
- Do not provide emergency treatment steps except telling user to contact doctor/emergency services.
- If this looks like a brand name, mention that exact composition may vary by product and user should check the product label.
- Keep it short.
- Use simple senior-friendly language.
- Include a clear safety note.
- Do not include any source line.
- Do not mention openFDA.

Return answer in this structure:

Medicine:
General Awareness:
Important Caution:
Safety Note:
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

        fallback_text = response.choices[0].message.content.strip()

        logger.info("LLM fallback medicine info completed")

        return remove_source_lines(fallback_text)

    except Exception as error:
        logger.error(f"LLM fallback medicine info failed: {error}")
        return ""
##


def build_medicine_base_answer(
    display_name: str,
    generic_name: str,
    common_use: str,
    notes: str,
    label: dict,
    query_type: str,
    source_status: str,
    second_medicine_name: str = ""
) -> str:
    """
    Build factual medicine answer from CSV/openFDA.

    This base answer is later formatted by LLM.
    """

    label_names = get_openfda_names(label) if label else {}

    if not generic_name:
        generic_name = label_names.get("generic_name", "")

    if not display_name:
        display_name = label_names.get("brand_name", "") or generic_name

    if not common_use:
        common_use = "Information not available from local alias file."

    uses = get_first_section(
        label,
        ["indications_and_usage", "purpose"]
    ) if label else ""

    active_ingredient = get_first_section(
        label,
        ["active_ingredient"]
    ) if label else ""

    warnings = get_first_section(
        label,
        [
            "boxed_warning",
            "warnings",
            "do_not_use",
            "ask_doctor",
            "ask_doctor_or_pharmacist",
            "stop_use"
        ]
    ) if label else ""

    side_effects = get_first_section(
        label,
        ["adverse_reactions", "stop_use"]
    ) if label else ""

    interactions = get_first_section(
        label,
        ["drug_interactions", "ask_doctor_or_pharmacist"]
    ) if label else ""

    pregnancy = get_first_section(
        label,
        ["pregnancy", "pregnancy_or_breast_feeding", "nursing_mothers"]
    ) if label else ""

    overdose = get_first_section(
        label,
        ["overdosage"]
    ) if label else ""

    storage = get_first_section(
        label,
        ["storage_and_handling", "storage"]
    ) if label else ""

    safe_uses = simplify_text(uses, max_sentences=2)
    safe_warnings = simplify_text(warnings, max_sentences=2)
    safe_side_effects = simplify_text(side_effects, max_sentences=2)
    safe_interactions = simplify_text(interactions, max_sentences=2)
    safe_pregnancy = simplify_text(pregnancy, max_sentences=2)
    safe_overdose = simplify_text(overdose, max_sentences=2)
    safe_storage = simplify_text(storage, max_sentences=2)

    if query_type == "dosage":
        focused_note = """
Dosage:
Young60 cannot provide a personal dosage recommendation.
Please follow your doctor's prescription or the product label.
For children, seniors, pregnancy, kidney/liver issues, or multiple medicines, ask a doctor/pharmacist.
"""
    elif query_type == "interactions":
        focused_note = f"""
Interaction Check:
User asked about combining medicines.
Primary medicine: {display_name}
Second medicine: {second_medicine_name if second_medicine_name else "Not clearly identified"}

Available interaction information:
{safe_interactions}

Important:
Medicine combinations should be confirmed with a doctor/pharmacist.
"""
    elif query_type == "side_effects":
        focused_note = f"""
Possible Side Effects:
{safe_side_effects}
"""
    elif query_type == "warnings":
        focused_note = f"""
Important Warnings / Precautions:
{safe_warnings}
"""
    elif query_type == "ingredients":
        focused_note = f"""
Ingredients / Active Ingredient:
{active_ingredient if active_ingredient else "Information not available from the connected source."}
"""
    elif query_type == "pregnancy":
        focused_note = f"""
Pregnancy / Breastfeeding:
{safe_pregnancy}

Important:
Please ask a doctor before using any medicine during pregnancy or breastfeeding.
"""
    elif query_type == "overdose":
        focused_note = f"""
Overdose Information:
{safe_overdose}

Emergency Note:
If overdose is suspected, contact emergency medical services or a doctor immediately.
"""
    elif query_type == "storage":
        focused_note = f"""
Storage:
{safe_storage}
"""
    else:
        focused_note = f"""
Used For:
Local/Common Use:
{common_use}

Label Use / Purpose:
{safe_uses}
"""

    return f"""
💊 Medicine: {display_name}

Generic Name:
{generic_name if generic_name else "Information not available."}

{focused_note}

General Important Caution:
{safe_warnings}

Possible Side Effects:
{safe_side_effects}

Notes:
{notes if notes else "Please check the product label and confirm with a doctor/pharmacist."}

Safety Note:
This is general awareness information only.
Please do not start, stop, or change medicine dosage without consulting a doctor.
If symptoms are serious, unusual, or worsening, please contact a doctor.

"""


# ==================================================
# Final LLM Formatter
# ==================================================
#helper
def remove_source_lines(text: str) -> str:
    """
    Remove source/debug lines from user-visible response.
    Source details should remain only in logs.
    """

    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    skip_next = False

    for line in lines:
        stripped = line.strip().lower()

        if skip_next:
            skip_next = False
            continue

        if stripped.startswith("source:"):
            skip_next = True
            continue

        if stripped.startswith("internal"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
##

def format_medicine_answer_with_llm(
    user_query: str,
    base_answer: str
) -> str:
    """
    Use LLM only to format the final medicine answer.

    Important:
    - Medicine data comes from CSV/openFDA/controlled fallback.
    - LLM must not invent dosage or personal medical advice.
    - Source/debug details must not be shown to the user.
    """

    if client is None:
        logger.warning("Final medicine formatting skipped because OPENAI_API_KEY is missing")
        return remove_source_lines(base_answer)

    logger.info("LLM final medicine formatter started")

    prompt = f"""
You are formatting a medicine response for Young60, an assistant for senior citizens.

Rules:
- Use only the medicine information provided below.
- Do not invent personal dosage advice.
- Do not tell the user to start, stop, combine, or change any medicine.
- Always keep the safety note.
- Do not show source/debug/internal lines.
- Remove any line starting with Source.
- Follow the user's requested format if mentioned.
- If user asks points/bullets/list, use clear bullet points.
- If user asks table, use a markdown table plus safety note below it.
- If user asks short summary, keep it short.
- If user asks WhatsApp style, make it copy-paste friendly.
- If user asks Hindi/Hinglish, respond in that style.
- Keep it simple and senior-friendly.

User query:
{user_query}

Medicine data response to format:
{base_answer}

Return only the final formatted answer.
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

        final_answer = response.choices[0].message.content.strip()

        final_answer = remove_source_lines(final_answer)

        logger.info("LLM final medicine formatter completed")

        return final_answer

    except Exception as error:
        logger.error(f"LLM final medicine formatter failed: {error}")
        return remove_source_lines(base_answer)

# ==================================================
# Fallback Responses
# ==================================================

def format_not_found_response(query: str) -> str:
    """
    Safe fallback response when medicine information is not found.
    """

    base_answer = f"""
I could not find reliable medicine information for:

{query}

Please check the spelling or ask a doctor/pharmacist.

Safety Note:
Do not take any medicine only based on app information.
Please consult a doctor for dosage, side effects, medicine combinations, pregnancy, allergy, or existing medical conditions.

Source:
No reliable medicine source found.
"""

    return format_medicine_answer_with_llm(
        user_query=query,
        base_answer=base_answer
    )


def format_alias_file_missing_response() -> str:
    """
    Response when alias file is missing.
    """

    return """
I could not find the medicine alias file.

Please check:
data/medicine_aliases.csv
"""


# ==================================================
# Main Medicine Service
# ==================================================

def medicine_help(query: str) -> str:
    """
    Main medicine service for Young60.

    Flow:
    1. Load local medicine_aliases.csv
    2. Parse user query using LLM + fallback rules
    3. Search alias CSV
    4. Fetch openFDA label using generic/medicine name
    5. Build factual safe answer
    6. Use LLM only for final formatting
    """

    logger.info("=" * 60)
    logger.info("Medicine service started")
    logger.info(f"User query: {query}")
    logger.info("=" * 60)

    alias_df = load_alias_data()

    if alias_df.empty:
        logger.error("Medicine alias data is empty or missing")
        return format_alias_file_missing_response()

    parsed_query = parse_medicine_query(query, alias_df)

    medicine_name = parsed_query.get("medicine_name", "")
    second_medicine_name = parsed_query.get("second_medicine_name", "")
    query_type = parsed_query.get("query_type", "general")
    response_format = parsed_query.get("response_format", "text")

    logger.info(f"Parsed medicine name: {medicine_name}")
    logger.info(f"Parsed second medicine name: {second_medicine_name}")
    logger.info(f"Medicine query type: {query_type}")
    logger.info(f"Response format: {response_format}")

    search_text = medicine_name if medicine_name else query

    medicine = find_medicine_alias(search_text, alias_df)

    # --------------------------------------------------
    # Case 1: Medicine found in local alias CSV
    # --------------------------------------------------

    if medicine is not None:
        logger.info("Medicine path: Local alias match")

        display_name = str(medicine["brand_name"]).strip()
        generic_name = str(medicine["generic_name"]).strip()
        common_use = str(medicine["common_use"]).strip()
        notes = str(medicine["notes"]).strip()

        logger.info(f"Brand name: {display_name}")
        logger.info(f"Generic name: {generic_name}")

        label = fetch_openfda_label(generic_name)

        if label is None:
            logger.warning("openFDA data not available. Returning CSV-only response.")

            base_answer = build_medicine_base_answer(
                display_name=display_name,
                generic_name=generic_name,
                common_use=common_use,
                notes=notes,
                label={},
                query_type=query_type,
                source_status="Local medicine alias file only",
                second_medicine_name=second_medicine_name
            )

            return format_medicine_answer_with_llm(
                user_query=query,
                base_answer=base_answer
            )

        base_answer = build_medicine_base_answer(
            display_name=display_name,
            generic_name=generic_name,
            common_use=common_use,
            notes=notes,
            label=label,
            query_type=query_type,
            source_status="Local alias file + openFDA public drug label API",
            second_medicine_name=second_medicine_name
        )

        return format_medicine_answer_with_llm(
            user_query=query,
            base_answer=base_answer
        )

    # --------------------------------------------------
    # Case 2: Not found in CSV, use parsed medicine name directly
    # --------------------------------------------------

    logger.info("Medicine path: No local alias found")

    if not medicine_name:
        logger.warning("No medicine name identified")
        return format_not_found_response(query)

    label = fetch_openfda_label(medicine_name)

    if label is None:
        logger.warning("openFDA data not found for parsed medicine name")

        fallback_info = generate_llm_fallback_medicine_info(
            medicine_name=medicine_name,
            query_type=query_type,
            second_medicine_name=second_medicine_name
        )

        if fallback_info:
            return format_medicine_answer_with_llm(
                user_query=query,
                base_answer=fallback_info
            )

        base_answer = f"""
I understood the medicine name as:

{medicine_name}

But I could not find verified medicine information for it.

Please check the spelling or ask a doctor/pharmacist.

Safety Note:
This information is for general awareness only and should not replace medical advice.
Please do not start, stop, combine, or change medicine dosage without consulting a doctor.
"""

        return format_medicine_answer_with_llm(
            user_query=query,
            base_answer=base_answer
        )

    label_names = get_openfda_names(label)

    display_name = medicine_name
    generic_name = label_names.get("generic_name", medicine_name)

    base_answer = build_medicine_base_answer(
        display_name=display_name,
        generic_name=generic_name,
        common_use="Information not available from local alias file.",
        notes="Medicine was not found in local alias file. Information is based on connected public label source if available.",
        label=label,
        query_type=query_type,
        source_status="LLM medicine-name extraction + openFDA public drug label API",
        second_medicine_name=second_medicine_name
    )

    return format_medicine_answer_with_llm(
        user_query=query,
        base_answer=base_answer
    )