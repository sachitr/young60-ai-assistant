from pathlib import Path
import os
import re
import json
import math
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

from core.service_logger import get_service_logger


# ==================================================
# Environment Setup
# ==================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_NEARBY_LOCATION = os.getenv("DEFAULT_NEARBY_LOCATION", "Rohini, Delhi, India")
DEFAULT_RADIUS_METERS = int(os.getenv("NEARBY_DEFAULT_RADIUS_METERS", "3000"))
MAX_RESULTS = int(os.getenv("NEARBY_MAX_RESULTS", "5"))

YOUNG60_CONTACT = os.getenv("YOUNG60_CONTACT", "young60-learning-app")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==================================================
# Logger Setup
# ==================================================

logger = get_service_logger("nearby_service")


# ==================================================
# Place Type Configuration
# ==================================================

PLACE_TYPE_CONFIG = {
    "hospital": {
        "label": "Hospitals",
        "tags": [
            {"amenity": "hospital"}
        ]
    },
    "pharmacy": {
        "label": "Pharmacies / Chemists",
        "tags": [
            {"amenity": "pharmacy"}
        ]
    },
    "clinic": {
        "label": "Clinics",
        "tags": [
            {"amenity": "clinic"}
        ]
    },
    "doctor": {
        "label": "Doctors",
        "tags": [
            {"amenity": "doctors"}
        ]
    },
    "atm": {
        "label": "ATMs",
        "tags": [
            {"amenity": "atm"}
        ]
    },
    "bank": {
        "label": "Banks",
        "tags": [
            {"amenity": "bank"}
        ]
    },
    "police": {
        "label": "Police Stations",
        "tags": [
            {"amenity": "police"}
        ]
    },
    "metro_station": {
        "label": "Metro / Subway Stations",
        "tags": [
            {"railway": "station"},
            {"railway": "subway_entrance"},
            {"station": "subway"}
        ]
    },
    "grocery": {
        "label": "Grocery / Supermarket",
        "tags": [
            {"shop": "supermarket"},
            {"shop": "convenience"},
            {"shop": "grocery"}
        ]
    },
    "park": {
        "label": "Parks",
        "tags": [
            {"leisure": "park"}
        ]
    },
    "library": {
        "label": "Libraries",
        "tags": [
            {"amenity": "library"}
        ]
    },
    "restaurant": {
        "label": "Restaurants",
        "tags": [
            {"amenity": "restaurant"}
        ]
    }
}

ALLOWED_PLACE_TYPES = set(PLACE_TYPE_CONFIG.keys())


# ==================================================
# Basic Helpers
# ==================================================

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


def clean_text(value: str) -> str:
    """
    Clean simple text.
    """

    if not value:
        return ""

    cleaned = str(value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned


def clean_location_text(text: str) -> str:
    """
    Clean location text extracted from user query.
    Removes format/radius/service words from location.
    """

    if not text:
        return ""

    cleaned = clean_text(text)
    cleaned = cleaned.replace("?", " ").replace(".", " ").replace("!", " ").strip()
    cleaned_lower = cleaned.lower()

    stop_markers = [
        " in table",
        " table format",
        " in bullets",
        " bullet points",
        " points",
        " whatsapp",
        " list",
        " please",
        " show",
        " provide",
        " give",
        " within",
        " radius",
        " near me",
        " around me",
        " nearby me",
    ]

    for marker in stop_markers:
        marker_index = cleaned_lower.find(marker)

        if marker_index != -1:
            cleaned = cleaned[:marker_index].strip()
            cleaned_lower = cleaned.lower()

    leading_words = [
        "near ",
        "around ",
        "in ",
        "at ",
        "for ",
    ]

    for word in leading_words:
        if cleaned_lower.startswith(word):
            cleaned = cleaned[len(word):].strip()
            cleaned_lower = cleaned.lower()

    return cleaned.strip()


def safe_radius(value, default_value: int = DEFAULT_RADIUS_METERS) -> int:
    """
    Keep radius inside practical range.
    """

    try:
        radius = int(value)
    except Exception:
        return default_value

    if radius < 500:
        return 500

    if radius > 10000:
        return 10000

    return radius


def extract_radius_by_rules(query: str) -> int:
    """
    Extract radius from query if user asks:
    - within 2 km
    - within 500 meters
    """

    q = query.lower()

    km_match = re.search(r"(?:within|radius|around)\s+(\d+)\s*(?:km|kilometer|kilometers)", q)

    if km_match:
        return safe_radius(int(km_match.group(1)) * 1000)

    meter_match = re.search(r"(?:within|radius|around)\s+(\d+)\s*(?:m|meter|meters)", q)

    if meter_match:
        return safe_radius(int(meter_match.group(1)))

    return DEFAULT_RADIUS_METERS


def haversine_distance_meters(lat1, lon1, lat2, lon2) -> int:
    """
    Calculate distance between two coordinates in meters.
    """

    radius_earth = 6371000

    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))

    delta_phi = math.radians(float(lat2) - float(lat1))
    delta_lambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return int(radius_earth * c)


def build_headers() -> dict:
    """
    Build headers for OSM/Nominatim calls.
    Nominatim requires app-identifying User-Agent.
    """

    return {
        "User-Agent": f"Young60/0.1 ({YOUNG60_CONTACT})"
    }


def ensure_osm_attribution(text: str) -> str:
    """
    Keep OpenStreetMap attribution visible.
    """

    if "OpenStreetMap" in text:
        return text

    return text.strip() + "\n\nMap data: © OpenStreetMap contributors"


# ==================================================
# Nearby Query Parser
# ==================================================

def parse_nearby_query_with_llm(query: str) -> dict:
    """
    Use LLM to parse user's nearby query into structured JSON.

    LLM understands query.
    APIs fetch real nearby places.
    """

    logger.info("LLM nearby query parser started")
    logger.info(f"Parser input query: {query}")

    default_result = {
        "place_type": "hospital",
        "target_place_name": "",
        "location": "",
        "radius_meters": DEFAULT_RADIUS_METERS,
        "response_format": "text",
        "urgency": "normal"
    }

    if client is None:
        logger.warning("LLM nearby parser skipped because OPENAI_API_KEY is missing")
        return default_result

    prompt = f"""
You are a nearby-place query parser for Young60, an assistant for senior citizens.

Convert the user's nearby-place question into JSON only.

Allowed place_type values:
- hospital
- pharmacy
- clinic
- doctor
- atm
- bank
- police
- metro_station
- grocery
- park
- restaurant
- library

Allowed response_format values:
- text
- table

Allowed urgency values:
- normal
- urgent

Rules:
- If user says chemist or medical store, use pharmacy.
- If user says doctor, physician, doctor clinic, use doctor.
- If user says police station, use police.
- If user says metro, subway, station, use metro_station.
- If user says grocery, supermarket, kirana, use grocery.
- If user says library, libraries, reading room, use library.
- If user says emergency, urgent, severe, accident, chest pain, breathing problem, use urgency urgent.
- If user says near me, around me, nearby me, keep location as empty string.
- If user gives a location, extract only the location.
- If user asks about a specific place/brand/name like Max Hospital, Apollo Hospital, HDFC Bank, put it in target_place_name.
- If user asks generic query like find hospital near Rohini, target_place_name should be empty.
- If user asks table, response_format should be table.
- If user asks bullets/list/points/WhatsApp note, response_format should be text.
- If user asks within N km, convert to meters.
- If no radius is mentioned, use {DEFAULT_RADIUS_METERS}.
- Return valid JSON only. No markdown. No explanation.

Examples:

User: Find hospitals near Rohini in table
JSON:
{{"place_type":"hospital","target_place_name":"","location":"Rohini, Delhi, India","radius_meters":3000,"response_format":"table","urgency":"normal"}}

User: Is Max Hospital near Rohini Delhi?
JSON:
{{"place_type":"hospital","target_place_name":"Max Hospital","location":"Rohini, Delhi, India","radius_meters":3000,"response_format":"text","urgency":"normal"}}

User: Chemist near me
JSON:
{{"place_type":"pharmacy","target_place_name":"","location":"","radius_meters":3000,"response_format":"text","urgency":"normal"}}

User: ATM around Connaught Place within 2 km
JSON:
{{"place_type":"atm","target_place_name":"","location":"Connaught Place, Delhi, India","radius_meters":2000,"response_format":"text","urgency":"normal"}}

User: Emergency hospital near Noida
JSON:
{{"place_type":"hospital","target_place_name":"","location":"Noida, India","radius_meters":3000,"response_format":"text","urgency":"urgent"}}

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
        logger.info(f"Raw nearby parser response: {raw_text}")

        parsed = safe_json_loads(raw_text)

        place_type = clean_text(parsed.get("place_type", "hospital")).lower()
        target_place_name = clean_text(parsed.get("target_place_name", ""))
        location = clean_location_text(parsed.get("location", ""))
        radius_meters = safe_radius(parsed.get("radius_meters", DEFAULT_RADIUS_METERS))
        response_format = clean_text(parsed.get("response_format", "text")).lower()
        urgency = clean_text(parsed.get("urgency", "normal")).lower()

        if place_type not in ALLOWED_PLACE_TYPES:
            place_type = "hospital"

        if response_format not in {"text", "table"}:
            response_format = "text"

        if urgency not in {"normal", "urgent"}:
            urgency = "normal"

        result = {
            "place_type": place_type,
            "target_place_name": target_place_name,
            "location": location,
            "radius_meters": radius_meters,
            "response_format": response_format,
            "urgency": urgency
        }

        logger.info(f"Parsed nearby query: {result}")

        return result

    except Exception as error:
        logger.error(f"LLM nearby parser failed: {error}")
        return default_result

def detect_place_type_by_rules(query: str) -> str:
    """
    Fallback place-type detection.
    """

    q = query.lower()

    if any(word in q for word in ["chemist", "pharmacy", "medical store", "medicine shop"]):
        return "pharmacy"

    if any(word in q for word in ["hospital", "emergency"]):
        return "hospital"

    if any(word in q for word in ["clinic"]):
        return "clinic"

    if any(word in q for word in ["doctor", "physician"]):
        return "doctor"

    if "atm" in q:
        return "atm"

    if "bank" in q:
        return "bank"

    if "police" in q:
        return "police"

    if any(word in q for word in ["metro", "subway", "station"]):
        return "metro_station"

    if any(word in q for word in ["grocery", "supermarket", "kirana"]):
        return "grocery"

    if "park" in q:
        return "park"
    
    if any(word in q for word in ["library", "libraries", "reading room"]):
        return "library"

    if any(word in q for word in ["restaurant", "food", "eat"]):
        return "restaurant"

    return "hospital"


def extract_location_by_rules(query: str) -> str:
    """
    Fallback location extraction.
    """

    q = query.lower()

    if any(word in q for word in ["near me", "around me", "nearby me"]):
        return ""

    patterns = [
        r"(?:near|around|in|at)\s+(.+)",
        r"(?:nearby)\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)

        if match:
            location = clean_location_text(match.group(1))

            if location:
                return location

    return ""


def parse_nearby_query_by_rules(query: str) -> dict:
    """
    Fallback parser if LLM fails.
    """

    logger.info("Rule-based nearby query parser started")

    q = query.lower()

    response_format = "text"

    if any(word in q for word in ["table", "table format"]):
        response_format = "table"

    urgency = "normal"

    if any(word in q for word in ["emergency", "urgent", "severe", "accident", "chest pain", "breathing"]):
        urgency = "urgent"

    target_place_name = ""

    known_place_patterns = [
        r"is\s+(.+?)\s+near",
        r"find\s+(.+?)\s+near",
        r"show\s+(.+?)\s+near",
    ]

    for pattern in known_place_patterns:
        match = re.search(pattern, query, re.IGNORECASE)

        if match:
            possible_target = clean_text(match.group(1))

            generic_words = [
                "hospital",
                "hospitals",
                "pharmacy",
                "chemist",
                "atm",
                "bank",
                "doctor",
                "clinic",
                "library",
                "restaurant",
                "park",
                "police station",
            ]

            if possible_target.lower() not in generic_words:
                target_place_name = possible_target

            break

    result = {
        "place_type": detect_place_type_by_rules(query),
        "target_place_name": target_place_name,
        "location": extract_location_by_rules(query),
        "radius_meters": extract_radius_by_rules(query),
        "response_format": response_format,
        "urgency": urgency
    }

    logger.info(f"Rule parser result: {result}")

    return result

def parse_nearby_query(query: str) -> dict:
    """
    Main nearby query parser.
    """

    llm_result = parse_nearby_query_with_llm(query)
    rule_result = parse_nearby_query_by_rules(query)

    final_result = llm_result.copy()

    if not final_result.get("target_place_name") and rule_result.get("target_place_name"):
        final_result["target_place_name"] = rule_result["target_place_name"]

    if not final_result.get("location") and rule_result.get("location"):
        final_result["location"] = rule_result["location"]

    if rule_result.get("response_format") == "table":
        final_result["response_format"] = "table"

    if rule_result.get("urgency") == "urgent":
        final_result["urgency"] = "urgent"

    if not final_result.get("radius_meters"):
        final_result["radius_meters"] = rule_result["radius_meters"]

    logger.info(f"Final parsed nearby query: {final_result}")

    return final_result

# ==================================================
# Geocoding
# ==================================================

def geocode_location(location: str):
    """
    Convert location name into latitude/longitude using Nominatim.
    """

    logger.info(f"Geocoding started for nearby location: {location}")

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": location,
        "format": "json",
        "limit": 1,
        "addressdetails": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=build_headers(),
            timeout=15
        )

        logger.info(f"Nominatim geocoding status: {response.status_code}")

        if response.status_code != 200:
            return None

        data = response.json()

        if not data:
            logger.warning(f"No geocoding result found for: {location}")
            return None

        place = data[0]

        result = {
            "display_name": place.get("display_name", location),
            "latitude": float(place.get("lat")),
            "longitude": float(place.get("lon"))
        }

        logger.info(f"Nominatim geocoding result: {result}")

        return result

    except requests.RequestException as error:
        logger.error(f"Nominatim request failed: {error}")
        return None

    except Exception as error:
        logger.error(f"Unexpected geocoding error: {error}")
        return None

def geocode_specific_place(place_name: str, location_hint: str):
    """
    Search a specific place/brand using Nominatim.
    Example: Max Hospital near Rohini Delhi
    """

    if not place_name:
        return None

    search_queries = [
        f"{place_name}, {location_hint}",
        place_name
    ]

    url = "https://nominatim.openstreetmap.org/search"

    for search_query in search_queries:
        try:
            response = requests.get(
                url,
                params={
                    "q": search_query,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1
                },
                headers=build_headers(),
                timeout=15
            )

            logger.info(f"Specific place geocoding status: {response.status_code}")

            if response.status_code != 200:
                continue

            data = response.json()

            if not data:
                continue

            place = data[0]

            return {
                "name": place.get("display_name", place_name),
                "latitude": float(place.get("lat")),
                "longitude": float(place.get("lon")),
            }

        except Exception as error:
            logger.error(f"Specific place geocoding failed: {error}")

    return None

# ==================================================
# Overpass Nearby Search
# ==================================================

def build_overpass_query(
    place_type: str,
    latitude: float,
    longitude: float,
    radius_meters: int
) -> str:
    """
    Build Overpass query for nearby POIs.
    """

    config = PLACE_TYPE_CONFIG.get(place_type, PLACE_TYPE_CONFIG["hospital"])
    tag_conditions = config["tags"]

    query_lines = []

    for condition in tag_conditions:
        filter_text = ""

        for key, value in condition.items():
            filter_text += f'["{key}"="{value}"]'

        query_lines.append(
            f'node{filter_text}(around:{radius_meters},{latitude},{longitude});'
        )
        query_lines.append(
            f'way{filter_text}(around:{radius_meters},{latitude},{longitude});'
        )
        query_lines.append(
            f'relation{filter_text}(around:{radius_meters},{latitude},{longitude});'
        )

    joined_queries = "\n  ".join(query_lines)

    return f"""
[out:json][timeout:25];
(
  {joined_queries}
);
out center tags;
"""


def call_overpass_api(query: str):
    """
    Call Overpass API with retry.
    """

    url = "https://overpass-api.de/api/interpreter"
    retry_status_codes = {429, 502, 503, 504}

    for attempt in range(1, 4):
        try:
            logger.info(f"Overpass API attempt {attempt}/3")

            response = requests.post(
                url,
                data={"data": query},
                headers=build_headers(),
                timeout=30
            )

            logger.info(f"Overpass API status: {response.status_code}")

            if response.status_code == 200:
                return response.json()

            if response.status_code in retry_status_codes:
                wait_seconds = attempt * 2
                logger.warning(
                    f"Overpass temporary issue {response.status_code}. Retrying after {wait_seconds} seconds."
                )
                time.sleep(wait_seconds)
                continue

            logger.warning(f"Overpass non-retryable status: {response.status_code}")
            return None

        except requests.RequestException as error:
            wait_seconds = attempt * 2
            logger.error(
                f"Overpass request failed: {error}. Retrying after {wait_seconds} seconds."
            )
            time.sleep(wait_seconds)

        except Exception as error:
            logger.error(f"Unexpected Overpass error: {error}")
            return None

    logger.warning("Overpass API failed after retries")
    return None


def extract_place_from_element(element: dict, user_lat: float, user_lon: float) -> dict:
    """
    Extract useful place information from OSM element.
    """

    tags = element.get("tags", {})

    if element.get("type") == "node":
        lat = element.get("lat")
        lon = element.get("lon")
    else:
        center = element.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")

    if lat is None or lon is None:
        return {}

    name = (
        tags.get("name")
        or tags.get("brand")
        or tags.get("operator")
        or "Name not available"
    )

    address_parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:suburb", ""),
        tags.get("addr:city", ""),
        tags.get("addr:postcode", "")
    ]

    address = ", ".join([part for part in address_parts if part])

    phone = (
        tags.get("phone")
        or tags.get("contact:phone")
        or tags.get("contact:mobile")
        or ""
    )

    opening_hours = tags.get("opening_hours", "")
    website = tags.get("website") or tags.get("contact:website") or ""

    distance_m = haversine_distance_meters(
        user_lat,
        user_lon,
        lat,
        lon
    )

    return {
        "name": name,
        "distance_m": distance_m,
        "address": address if address else "Address not available",
        "phone": phone if phone else "Phone not available",
        "opening_hours": opening_hours if opening_hours else "Opening hours not available",
        "website": website if website else "",
        "latitude": lat,
        "longitude": lon
    }


def search_nearby_places(
    place_type: str,
    latitude: float,
    longitude: float,
    radius_meters: int
) -> list:
    """
    Search nearby places using Overpass API.
    """

    logger.info(
        f"Nearby search started | type={place_type}, lat={latitude}, lon={longitude}, radius={radius_meters}"
    )

    query = build_overpass_query(
        place_type=place_type,
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters
    )

    data = call_overpass_api(query)

    if data is None:
        return []

    elements = data.get("elements", [])

    places = []

    seen_keys = set()

    for element in elements:
        place = extract_place_from_element(
            element=element,
            user_lat=latitude,
            user_lon=longitude
        )

        if not place:
            continue

        dedupe_key = (
            place.get("name", "").lower(),
            round(float(place.get("latitude", 0)), 5),
            round(float(place.get("longitude", 0)), 5)
        )

        if dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        places.append(place)

    places = sorted(places, key=lambda item: item["distance_m"])

    logger.info(f"Nearby places found: {len(places)}")

    return places[:MAX_RESULTS]


# ==================================================
# Response Builders
# ==================================================

def build_places_base_answer(
    place_type: str,
    location: str,
    radius_meters: int,
    places: list,
    response_format: str,
    urgency: str,
    specific_place_note: str = ""
) -> str:
    """
    Build factual nearby places answer from OSM data.
    """

    label = PLACE_TYPE_CONFIG.get(place_type, PLACE_TYPE_CONFIG["hospital"])["label"]

    if not places:
        return f"""
I could not find nearby {label.lower()} for:

{location}

Search radius:
{radius_meters} meters

{specific_place_note}

Please try a bigger area or a clearer location.

Safety Note:
If this is an emergency, please call local emergency services immediately.
"""

    emergency_note = ""

    if urgency == "urgent":
        emergency_note = """
Urgent Note:
If this is a medical or safety emergency, please call local emergency services immediately.
Do not wait only for app results.
"""

    if response_format == "table":
        rows = []

        for index, place in enumerate(places, start=1):
            rows.append(
                f"| {index} | {place['name']} | {place['distance_m']} m | "
                f"{place['address']} | {place['phone']} | {place['opening_hours']} |"
            )

        table_text = "\n".join(rows)

        return f"""
📍 Nearby {label} near {location}

Search radius:
{radius_meters} meters

{specific_place_note}

| No. | Name | Distance | Address | Phone | Opening Hours |
|---:|---|---:|---|---|---|
{table_text}

{emergency_note}
Senior Tip:
Please call before visiting if phone number is available.
Map data: © OpenStreetMap contributors
"""

    bullet_lines = []

    for index, place in enumerate(places, start=1):
        bullet_lines.append(
            f"""
{index}. {place['name']}
   - Distance: {place['distance_m']} meters
   - Address: {place['address']}
   - Phone: {place['phone']}
   - Opening Hours: {place['opening_hours']}
"""
        )

    places_text = "\n".join(bullet_lines)

    return f"""
📍 Nearby {label} near {location}

Search radius:
{radius_meters} meters

{specific_place_note}

{places_text}

{emergency_note}
Senior Tip:
Please call before visiting if phone number is available.
Map data: © OpenStreetMap contributors
"""


def format_nearby_answer_with_llm(
    user_query: str,
    base_answer: str
) -> str:
    """
    Use LLM only to format final nearby answer.

    Places come from OSM/Overpass.
    LLM must not invent new places.
    """

    if client is None:
        logger.warning("Final nearby formatting skipped because OPENAI_API_KEY is missing")
        return ensure_osm_attribution(base_answer)

    logger.info("LLM final nearby formatter started")

    prompt = f"""
You are formatting a nearby places response for Young60, an assistant for senior citizens.

Rules:
- Use only the places provided below.
- Do not invent any place, phone number, distance, address, rating, or opening hours.
- Keep it simple and senior-friendly.
- Follow the user's requested format if mentioned.
- If user asks table, use markdown table.
- If user asks points/bullets/list, use clear bullets.
- If user asks short, keep it short.
- If user asks WhatsApp style, make it copy-paste friendly.
- If user asks Hindi/Hinglish, respond in that style.
- Keep emergency/safety note if present.
- Keep this attribution exactly once: Map data: © OpenStreetMap contributors
- Do not show internal/debug fields.

User query:
{user_query}

Nearby places data:
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
        final_answer = ensure_osm_attribution(final_answer)

        logger.info("LLM final nearby formatter completed")

        return final_answer

    except Exception as error:
        logger.error(f"LLM final nearby formatter failed: {error}")
        return ensure_osm_attribution(base_answer)


def format_geocoding_failed_response(location: str) -> str:
    """
    Fallback response when location cannot be geocoded.
    """

    return f"""
I could not understand this location clearly:

{location}

Please try with city/area name.

Examples:
- Hospital near Rohini
- Pharmacy near Connaught Place
- ATM near Noida Sector 18
- Police station near Jaipur

Safety Note:
If this is an emergency, please call local emergency services immediately.
"""


# ==================================================
# Main Nearby Service
# ==================================================

def nearby_help(query: str, user_location: dict = None) -> str:
    """
    Main nearby service for Young60.

    Flow:
    1. Parse nearby query using LLM + fallback rules
    2. Resolve location using Nominatim
    3. Search POIs using Overpass API
    4. Build factual answer
    5. Format final answer with LLM
    """

    logger.info("=" * 60)
    logger.info("Nearby service started")
    logger.info(f"User query: {query}")
    logger.info("=" * 60)

    parsed_query = parse_nearby_query(query)

    place_type = parsed_query.get("place_type", "hospital")
    target_place_name = parsed_query.get("target_place_name", "")
    location = parsed_query.get("location", "")
    radius_meters = safe_radius(parsed_query.get("radius_meters", DEFAULT_RADIUS_METERS))
    response_format = parsed_query.get("response_format", "text")
    urgency = parsed_query.get("urgency", "normal")

    location_source = "query"

    logger.info(f"Nearby place type: {place_type}")
    logger.info(f"Target place name: {target_place_name}")
    logger.info(f"Nearby parsed location: {location}")
    logger.info(f"Radius meters: {radius_meters}")
    logger.info(f"Response format: {response_format}")
    logger.info(f"Urgency: {urgency}")

    # --------------------------------------------------
    # Use browser/device location for "near me"
    # --------------------------------------------------

    if not location and user_location:
        try:
            latitude = float(user_location.get("latitude"))
            longitude = float(user_location.get("longitude"))

            location_source = "browser_location"
            display_location = "your current location"

            logger.info("Using browser current location")
            logger.info(f"User latitude: {latitude}")
            logger.info(f"User longitude: {longitude}")

        except Exception as error:
            logger.error(f"Invalid browser location received: {error}")
            latitude = None
            longitude = None
            display_location = ""

    else:
        latitude = None
        longitude = None
        display_location = ""

    # --------------------------------------------------
    # If browser location not available, use query/default location
    # --------------------------------------------------

    if latitude is None or longitude is None:
        if not location:
            location = DEFAULT_NEARBY_LOCATION
            location_source = "default"

        logger.info(f"Nearby location for geocoding: {location}")
        logger.info(f"Location source: {location_source}")

        location_info = geocode_location(location)

        if location_info is None:
            logger.warning("Nearby geocoding failed")
            return format_geocoding_failed_response(location)

        latitude = location_info.get("latitude")
        longitude = location_info.get("longitude")
        display_location = location_info.get("display_name", location)

    specific_place_note = ""

    if target_place_name:
        specific_place = geocode_specific_place(
            place_name=target_place_name,
            location_hint=location
        )

        if specific_place:
            distance_m = haversine_distance_meters(
                latitude,
                longitude,
                specific_place["latitude"],
                specific_place["longitude"]
            )

            distance_km = round(distance_m / 1000, 2)

            if distance_m <= radius_meters:
                specific_place_note = f"""
Specific Place Check:
{target_place_name} appears to be around {distance_km} km from {display_location}.
It is within the selected search radius of {round(radius_meters / 1000, 1)} km.
"""
            else:
                specific_place_note = f"""
Specific Place Check:
{target_place_name} appears to be around {distance_km} km from {display_location}.
It is not very close based on the selected search radius of {round(radius_meters / 1000, 1)} km.
"""
        else:
            specific_place_note = f"""
Specific Place Check:
I could not reliably locate {target_place_name} from map data.
Below are nearby {PLACE_TYPE_CONFIG.get(place_type, PLACE_TYPE_CONFIG["hospital"])["label"].lower()} found around the requested area.
"""

    places = search_nearby_places(
        place_type=place_type,
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters
    )

    location_display_for_user = display_location

    if location_source == "default":
        location_display_for_user = (
            f"{display_location}\n\n"
            f"Note: No exact location was provided, so I used default location: {DEFAULT_NEARBY_LOCATION}"
        )

    if location_source == "browser_location":
        location_display_for_user = "your current location"

    base_answer = build_places_base_answer(
        place_type=place_type,
        location=location_display_for_user,
        radius_meters=radius_meters,
        places=places,
        response_format=response_format,
        urgency=urgency,
        specific_place_note=specific_place_note
    )

    return format_nearby_answer_with_llm(
        user_query=query,
        base_answer=base_answer
    )