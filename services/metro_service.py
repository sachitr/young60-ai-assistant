import os
import re
import json
import requests

from dotenv import load_dotenv
from openai import OpenAI

from core.service_logger import get_service_logger


# ==================================================
# Environment Setup
# ==================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==================================================
# Logger Setup
# ==================================================

logger = get_service_logger("metro_service")


# ==================================================
# Basic Helpers
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


def remove_internal_lines(text: str) -> str:
    """
    Remove internal/source/debug lines from user response.
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

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def parse_google_duration(duration_value: str) -> str:
    """
    Convert Google duration like '2340s' into readable minutes.
    """

    if not duration_value:
        return "Time not available"

    try:
        seconds = int(str(duration_value).replace("s", ""))
        minutes = round(seconds / 60)

        if minutes < 60:
            return f"{minutes} minutes"

        hours = minutes // 60
        remaining_minutes = minutes % 60

        if remaining_minutes == 0:
            return f"{hours} hour(s)"

        return f"{hours} hour(s) {remaining_minutes} minutes"

    except Exception:
        return str(duration_value)


def meters_to_km(distance_meters) -> str:
    """
    Convert meters to readable distance.
    """

    try:
        meters = float(distance_meters)

        if meters < 1000:
            return f"{round(meters)} m"

        return f"{round(meters / 1000, 1)} km"

    except Exception:
        return "Distance not available"


# ==================================================
# Metro Query Parser
# ==================================================

def parse_metro_query_with_llm(query: str) -> dict:
    """
    Use LLM to parse the metro/public transport query.

    LLM only understands the query.
    Google Routes API calculates the real route.
    """

    logger.info("LLM metro query parser started")
    logger.info(f"Parser input query: {query}")

    default_result = {
        "source": "",
        "destination": "",
        "response_format": "text",
        "route_preference": "FEWER_TRANSFERS",
        "transit_mode": "SUBWAY",
        "need_accessibility_help": False
    }

    if client is None:
        logger.warning("LLM metro parser skipped because OPENAI_API_KEY missing")
        return default_result

    prompt = f"""
You are a route query parser for Young60, an assistant for senior citizens.

Convert the user's metro/public transport query into JSON only.

Allowed response_format values:
- text
- table

Allowed route_preference values:
- FEWER_TRANSFERS
- LESS_WALKING

Allowed transit_mode values:
- SUBWAY
- TRAIN
- RAIL
- BUS
- ANY

Rules:
- Extract source and destination as place/station names.
- If user asks specifically for metro/subway, transit_mode should be SUBWAY.
- If user asks train/railway, transit_mode should be TRAIN.
- If user asks public transport generally, transit_mode should be ANY.
- If user asks fewer changes/interchanges, route_preference should be FEWER_TRANSFERS.
- If user asks less walking/senior citizen/wheelchair, route_preference should be LESS_WALKING and need_accessibility_help true.
- If user asks table, response_format should be table.
- If user asks bullets/points/WhatsApp note, response_format should be text.
- Return valid JSON only. No markdown. No explanation.

Examples:

User: Metro route from Rohini West to AIIMS
JSON:
{{"source":"Rohini West","destination":"AIIMS","response_format":"text","route_preference":"FEWER_TRANSFERS","transit_mode":"SUBWAY","need_accessibility_help":false}}

User: How to go from Rajiv Chowk to Botanical Garden by metro in table
JSON:
{{"source":"Rajiv Chowk","destination":"Botanical Garden","response_format":"table","route_preference":"FEWER_TRANSFERS","transit_mode":"SUBWAY","need_accessibility_help":false}}

User: Route to AIIMS from Rohini West for senior citizen with less walking
JSON:
{{"source":"Rohini West","destination":"AIIMS","response_format":"text","route_preference":"LESS_WALKING","transit_mode":"SUBWAY","need_accessibility_help":true}}

User: public transport from Agra Railway Station to Taj Mahal
JSON:
{{"source":"Agra Railway Station","destination":"Taj Mahal","response_format":"text","route_preference":"FEWER_TRANSFERS","transit_mode":"ANY","need_accessibility_help":false}}

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
        logger.info(f"Raw metro parser response: {raw_text}")

        parsed = safe_json_loads(raw_text)

        source = clean_text(parsed.get("source", ""))
        destination = clean_text(parsed.get("destination", ""))
        response_format = clean_text(parsed.get("response_format", "text")).lower()
        route_preference = clean_text(parsed.get("route_preference", "FEWER_TRANSFERS")).upper()
        transit_mode = clean_text(parsed.get("transit_mode", "SUBWAY")).upper()
        need_accessibility_help = bool(parsed.get("need_accessibility_help", False))

        if response_format not in {"text", "table"}:
            response_format = "text"

        if route_preference not in {"FEWER_TRANSFERS", "LESS_WALKING"}:
            route_preference = "FEWER_TRANSFERS"

        if transit_mode not in {"SUBWAY", "TRAIN", "RAIL", "BUS", "ANY"}:
            transit_mode = "SUBWAY"

        result = {
            "source": source,
            "destination": destination,
            "response_format": response_format,
            "route_preference": route_preference,
            "transit_mode": transit_mode,
            "need_accessibility_help": need_accessibility_help
        }

        logger.info(f"Parsed metro query: {result}")

        return result

    except Exception as error:
        logger.error(f"LLM metro parser failed: {error}")
        return default_result


def parse_metro_query_by_rules(query: str) -> dict:
    """
    Rule-based fallback parser.
    """

    logger.info("Rule-based metro parser started")

    q = query.lower().strip()

    result = {
        "source": "",
        "destination": "",
        "response_format": "text",
        "route_preference": "FEWER_TRANSFERS",
        "transit_mode": "SUBWAY",
        "need_accessibility_help": False
    }

    if any(word in q for word in ["table", "table format"]):
        result["response_format"] = "table"

    if any(word in q for word in ["less walking", "senior", "wheelchair", "lift", "escalator"]):
        result["route_preference"] = "LESS_WALKING"
        result["need_accessibility_help"] = True

    if any(word in q for word in ["bus"]):
        result["transit_mode"] = "BUS"

    elif any(word in q for word in ["train", "railway"]):
        result["transit_mode"] = "TRAIN"

    elif any(word in q for word in ["public transport", "transit"]):
        result["transit_mode"] = "ANY"

    patterns = [
        r"from\s+(.+?)\s+to\s+(.+)",
        r"route\s+from\s+(.+?)\s+to\s+(.+)",
        r"to\s+(.+?)\s+from\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)

        if match:
            if pattern.startswith("to"):
                result["destination"] = clean_text(match.group(1))
                result["source"] = clean_text(match.group(2))
            else:
                result["source"] = clean_text(match.group(1))
                result["destination"] = clean_text(match.group(2))

            break

    logger.info(f"Rule parser result: {result}")

    return result


def parse_metro_query(query: str) -> dict:
    """
    Main metro query parser.
    """

    llm_result = parse_metro_query_with_llm(query)
    rule_result = parse_metro_query_by_rules(query)

    final_result = llm_result.copy()

    if not final_result.get("source") and rule_result.get("source"):
        final_result["source"] = rule_result["source"]

    if not final_result.get("destination") and rule_result.get("destination"):
        final_result["destination"] = rule_result["destination"]

    if rule_result.get("response_format") == "table":
        final_result["response_format"] = "table"

    if rule_result.get("need_accessibility_help"):
        final_result["need_accessibility_help"] = True
        final_result["route_preference"] = "LESS_WALKING"

    logger.info(f"Final parsed metro query: {final_result}")

    return final_result


# ==================================================
# Google Routes API
# ==================================================

def build_transit_preferences(transit_mode: str, route_preference: str) -> dict:
    """
    Build Google Routes API transitPreferences.
    """

    transit_preferences = {
        "routingPreference": route_preference
    }

    if transit_mode == "SUBWAY":
        transit_preferences["allowedTravelModes"] = ["SUBWAY"]

    elif transit_mode == "TRAIN":
        transit_preferences["allowedTravelModes"] = ["TRAIN"]

    elif transit_mode == "RAIL":
        transit_preferences["allowedTravelModes"] = ["RAIL", "TRAIN", "SUBWAY", "LIGHT_RAIL"]

    elif transit_mode == "BUS":
        transit_preferences["allowedTravelModes"] = ["BUS"]

    # ANY means no allowedTravelModes restriction.
    # Google will choose available public transport modes.

    return transit_preferences


def call_google_routes_api(
    source: str,
    destination: str,
    transit_mode: str,
    route_preference: str
):
    """
    Call Google Routes API using TRANSIT mode.
    """

    if not GOOGLE_MAPS_API_KEY:
        logger.error("GOOGLE_MAPS_API_KEY missing")
        return {
            "error": "GOOGLE_MAPS_API_KEY is missing"
        }

    logger.info("Google Routes API call started")
    logger.info(f"Origin: {source}")
    logger.info(f"Destination: {destination}")
    logger.info(f"Transit mode: {transit_mode}")
    logger.info(f"Route preference: {route_preference}")

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": (
            "routes.duration,"
            "routes.distanceMeters,"
            "routes.legs.steps"
        )
    }

    payload = {
        "origin": {
            "address": source
        },
        "destination": {
            "address": destination
        },
        "travelMode": "TRANSIT",
        "computeAlternativeRoutes": False,
        "languageCode": "en",
        "units": "METRIC",
        "transitPreferences": build_transit_preferences(
            transit_mode=transit_mode,
            route_preference=route_preference
        )
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"Google Routes API status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Google Routes API error: {response.text}")
            return {
                "error": response.text
            }

        return response.json()

    except requests.RequestException as error:
        logger.error(f"Google Routes API request failed: {error}")
        return {
            "error": str(error)
        }

    except Exception as error:
        logger.error(f"Unexpected Google Routes API error: {error}")
        return {
            "error": str(error)
        }


# ==================================================
# Route Extraction
# ==================================================

def extract_transit_line_name(transit_details: dict) -> str:
    """
    Extract transit line name from Google response.
    """

    transit_line = transit_details.get("transitLine", {})

    line_name = (
        transit_line.get("name")
        or transit_line.get("nameShort")
        or transit_line.get("nameLong")
        or ""
    )

    vehicle = transit_line.get("vehicle", {})
    vehicle_name = vehicle.get("name", "")
    vehicle_type = vehicle.get("type", "")

    if line_name:
        return line_name

    if vehicle_name:
        return vehicle_name

    if vehicle_type:
        return vehicle_type

    return "Transit line"


def extract_step_instruction(step: dict) -> dict:
    """
    Extract useful step details from one Google route step.
    """

    travel_mode = step.get("travelMode", "")
    localized_values = step.get("localizedValues", {})
    navigation_instruction = step.get("navigationInstruction", {})

    distance_text = (
        localized_values.get("distance", {}).get("text")
        or meters_to_km(step.get("distanceMeters", 0))
    )

    duration_text = (
        localized_values.get("staticDuration", {}).get("text")
        or localized_values.get("duration", {}).get("text")
        or ""
    )

    if travel_mode == "TRANSIT":
        transit_details = step.get("transitDetails", {})
        stop_details = transit_details.get("stopDetails", {})

        departure_stop = stop_details.get("departureStop", {}).get("name", "")
        arrival_stop = stop_details.get("arrivalStop", {}).get("name", "")

        line_name = extract_transit_line_name(transit_details)
        headsign = transit_details.get("headsign", "")
        stop_count = transit_details.get("stopCount", "")

        return {
            "type": "transit",
            "instruction": f"Take {line_name}",
            "line": line_name,
            "from": departure_stop,
            "to": arrival_stop,
            "headsign": headsign,
            "stop_count": stop_count,
            "distance": distance_text,
            "duration": duration_text
        }

    instruction = navigation_instruction.get("instructions", "")

    if not instruction:
        instruction = "Walk / transfer"

    return {
        "type": "walk",
        "instruction": instruction,
        "line": "",
        "from": "",
        "to": "",
        "headsign": "",
        "stop_count": "",
        "distance": distance_text,
        "duration": duration_text
    }


def extract_route_facts(api_data: dict) -> dict:
    """
    Extract route facts from Google Routes API response.
    """

    if not api_data:
        return {}

    if "error" in api_data:
        return {
            "error": api_data.get("error")
        }

    routes = api_data.get("routes", [])

    if not routes:
        return {}

    route = routes[0]

    duration = parse_google_duration(route.get("duration", ""))
    distance = meters_to_km(route.get("distanceMeters", 0))

    steps = []

    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            steps.append(extract_step_instruction(step))

    transit_steps = [step for step in steps if step["type"] == "transit"]
    walking_steps = [step for step in steps if step["type"] == "walk"]

    interchanges = []

    for index in range(len(transit_steps) - 1):
        interchanges.append(transit_steps[index].get("to", ""))

    facts = {
        "duration": duration,
        "distance": distance,
        "steps": steps,
        "transit_steps": transit_steps,
        "walking_steps": walking_steps,
        "interchanges": [item for item in interchanges if item]
    }

    logger.info(f"Extracted route facts: {facts}")

    return facts


# ==================================================
# Response Builder
# ==================================================

def build_metro_base_answer(
    source: str,
    destination: str,
    route_facts: dict,
    response_format: str,
    need_accessibility_help: bool
) -> str:
    """
    Build factual metro/public transport answer.
    """

    if not route_facts:
        return f"""
I could not find a confident transit route.

From:
{source}

To:
{destination}

Please check the place names and try again.
"""

    if "error" in route_facts:
        return f"""
I could not fetch the transit route right now.

From:
{source}

To:
{destination}

Technical issue:
{route_facts.get("error")}

Please check API key/billing or try again later.
"""

    duration = route_facts.get("duration", "Time not available")
    distance = route_facts.get("distance", "Distance not available")
    steps = route_facts.get("steps", [])
    interchanges = route_facts.get("interchanges", [])

    if response_format == "table":
        rows = []

        for index, step in enumerate(steps, start=1):
            if step["type"] == "transit":
                rows.append(
                    f"| {index} | Transit | {step['line']} | "
                    f"{step['from']} | {step['to']} | "
                    f"{step['headsign']} | {step['stop_count']} | "
                    f"{step['duration']} |"
                )
            else:
                rows.append(
                    f"| {index} | Walk/Transfer |  |  |  | "
                    f"{step['instruction']} |  | {step['duration']} |"
                )

        table_text = "\n".join(rows)

        base_answer = f"""
🚇 Transit Route: {source} to {destination}

Estimated Time:
{duration}

Total Distance:
{distance}

Interchanges:
{", ".join(interchanges) if interchanges else "No major interchange detected"}

| Step | Type | Line | From | To | Direction / Instruction | Stops | Duration |
|---:|---|---|---|---|---|---:|---|
{table_text}
"""

    else:
        step_lines = []

        for index, step in enumerate(steps, start=1):
            if step["type"] == "transit":
                step_text = (
                    f"{index}. Take {step['line']} from {step['from']} "
                    f"to {step['to']}."
                )

                if step.get("headsign"):
                    step_text += f" Direction: {step['headsign']}."

                if step.get("stop_count"):
                    step_text += f" Approx. stops: {step['stop_count']}."

                if step.get("duration"):
                    step_text += f" Time: {step['duration']}."

                step_lines.append(step_text)

            else:
                step_lines.append(
                    f"{index}. {step['instruction']} "
                    f"{'Time: ' + step['duration'] if step.get('duration') else ''}"
                )

        route_text = "\n".join(step_lines)

        base_answer = f"""
🚇 Transit Route: {source} to {destination}

Estimated Time:
{duration}

Total Distance:
{distance}

Interchanges:
{", ".join(interchanges) if interchanges else "No major interchange detected"}

Route:
{route_text}
"""

    if need_accessibility_help:
        base_answer += """

Senior / Accessibility Tip:
Please prefer lifts/escalators where available, avoid peak rush hours, and keep extra time for walking or interchange.
"""

    base_answer += """

Senior Tip:
Please verify live service status before travelling and avoid rushing during transfers.
"""

    return base_answer


def format_metro_answer_with_llm(
    user_query: str,
    base_answer: str
) -> str:
    """
    Use LLM only to format final metro/public transport response.
    LLM must not change route facts.
    """

    if client is None:
        logger.warning("Final metro formatter skipped because OPENAI_API_KEY missing")
        return remove_internal_lines(base_answer)

    logger.info("LLM final metro formatter started")

    prompt = f"""
You are formatting a transit/metro route response for Young60, an assistant for senior citizens.

Rules:
- Use only the route information provided below.
- Do not invent station names, lines, times, stops, or interchanges.
- Keep it simple and senior-friendly.
- If user asks table, use markdown table.
- If user asks bullets/points, use bullets.
- If user asks WhatsApp style, make it copy-paste friendly.
- If user asks Hindi/Hinglish, respond in that style.
- Keep senior/accessibility tips if present.
- Do not show internal/debug/source lines.

User query:
{user_query}

Route data:
{base_answer}

Return only final formatted answer.
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
        final_answer = remove_internal_lines(final_answer)

        logger.info("LLM final metro formatter completed")

        return final_answer

    except Exception as error:
        logger.error(f"LLM final metro formatter failed: {error}")
        return remove_internal_lines(base_answer)


def format_missing_input_response(query: str, source: str, destination: str) -> str:
    """
    Fallback if source/destination missing.
    """

    return f"""
I could not understand the route clearly.

Your query:
{query}

Detected source:
{source if source else "Not found"}

Detected destination:
{destination if destination else "Not found"}

Please try like:
- Metro route from Rohini West to AIIMS
- How to go from Rajiv Chowk to Botanical Garden by metro?
- Public transport from Agra Railway Station to Taj Mahal
"""


# ==================================================
# Main Metro Service
# ==================================================

def metro_help(query: str) -> str:
    """
    Main metro/public transport service for Young60.

    Flow:
    1. Parse query using LLM + fallback rules
    2. Call Google Routes API with TRANSIT mode
    3. Extract route facts from API response
    4. Build factual answer
    5. Use LLM only for final formatting
    """

    logger.info("=" * 60)
    logger.info("Metro service started")
    logger.info(f"User query: {query}")
    logger.info("=" * 60)

    parsed_query = parse_metro_query(query)

    source = parsed_query.get("source", "")
    destination = parsed_query.get("destination", "")
    response_format = parsed_query.get("response_format", "text")
    route_preference = parsed_query.get("route_preference", "FEWER_TRANSFERS")
    transit_mode = parsed_query.get("transit_mode", "SUBWAY")
    need_accessibility_help = parsed_query.get("need_accessibility_help", False)

    logger.info(f"Parsed source: {source}")
    logger.info(f"Parsed destination: {destination}")
    logger.info(f"Response format: {response_format}")
    logger.info(f"Route preference: {route_preference}")
    logger.info(f"Transit mode: {transit_mode}")
    logger.info(f"Need accessibility help: {need_accessibility_help}")

    if not source or not destination:
        return format_missing_input_response(
            query=query,
            source=source,
            destination=destination
        )

    api_data = call_google_routes_api(
        source=source,
        destination=destination,
        transit_mode=transit_mode,
        route_preference=route_preference
    )

    route_facts = extract_route_facts(api_data)

    base_answer = build_metro_base_answer(
        source=source,
        destination=destination,
        route_facts=route_facts,
        response_format=response_format,
        need_accessibility_help=need_accessibility_help
    )

    return format_metro_answer_with_llm(
        user_query=query,
        base_answer=base_answer
    )