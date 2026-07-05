from pathlib import Path
import os
import re
import json
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

from core.service_logger import get_service_logger


##helper functions
def clean_text(value: str) -> str:
    """
    Clean simple text value.
    """

    if not value:
        return ""

    cleaned = str(value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned

def clean_weather_location_text(text: str) -> str:
    """
    Clean location extracted from weather query.
    Removes forecast/time/format instruction words without damaging city names.
    """

    if not text:
        return ""

    cleaned = clean_text(text)

    stop_patterns = [
        r"\s+for\s+next\s+\d+\s+days?.*",
        r"\s+next\s+\d+\s+days?.*",
        r"\s+for\s+\d+\s+days?.*",
        r"\s+for\s+tomorrow.*",
        r"\s+tomorrow.*",
        r"\s+today.*",
        r"\s+tonight.*",
        r"\s+this\s+week.*",
        r"\s+next\s+week.*",
        r"\s+in\s+table\s+format.*",
        r"\s+table\s+format.*",
        r"\s+provide\s+your\s+response.*",
        r"\s+ignore\s+snowfall.*",
        r"\s+ignore\s+snow.*",
        r"\s+snowfall\s+not\s+required.*",
        r"\s+not\s+required.*",
        r"\s+please.*",
    ]

    for pattern in stop_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = cleaned.replace(".", " ").replace("?", " ").replace(",", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def safe_json_loads(raw_text: str) -> dict:
    """
    Safely parse JSON from LLM response.
    Handles plain JSON and markdown code block JSON.
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


##

# ==================================================
# Environment Setup
# ==================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_WEATHER_LOCATION = os.getenv("DEFAULT_WEATHER_LOCATION", "Delhi, India")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==================================================
# Logger Setup
# ==================================================

logger = get_service_logger("weather_service")


# ==================================================
# Basic Helpers
# ==================================================

def safe_int(value, default_value: int = 7, minimum: int = 1, maximum: int = 16) -> int:
    """
    Convert value to int safely and keep it inside allowed range.
    """

    try:
        number = int(value)
    except Exception:
        return default_value

    if number < minimum:
        return minimum

    if number > maximum:
        return maximum

    return number


def clean_location_text(text: str) -> str:
    """
    Clean location text extracted from user query.
    Removes time/format/request words from the location.
    """

    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("?", " ").replace(".", " ").replace("!", " ").strip()

    cleaned_lower = cleaned.lower()

    stop_markers = [
        " next week",
        " this week",
        " tomorrow",
        " today",
        " tonight",
        " weekend",
        " right now",
        " now",
        " currently",
        " please",
        " provide",
        " give",
        " show",
        " tell",
        " in table",
        " table format",
        " day-wise",
        " day wise",
        " date-wise",
        " date wise",
    ]

    for marker in stop_markers:
        marker_index = cleaned_lower.find(marker)

        if marker_index != -1:
            cleaned = cleaned[:marker_index].strip()
            cleaned_lower = cleaned.lower()

    leading_words = [
        "in ",
        "for ",
        "at ",
        "of ",
    ]

    for word in leading_words:
        if cleaned_lower.startswith(word):
            cleaned = cleaned[len(word):].strip()
            cleaned_lower = cleaned.lower()

    return cleaned.strip()


def is_valid_location(location: str) -> bool:
    """
    Basic validation to avoid treating weather words as location.
    """

    if not location:
        return False

    location_lower = location.lower().strip()

    invalid_values = {
        "weather",
        "temperature",
        "mausam",
        "forecast",
        "rain",
        "snow",
        "storm",
        "hot",
        "heat",
        "today",
        "aaj",
        "tomorrow",
        "kal",
        "next week",
        "this week",
        "now",
        "right now",
        "current",
        "currently",
    }

    if location_lower in invalid_values:
        return False

    if len(location_lower) < 2:
        return False

    return True


# ==================================================
# Weather Query Parser
# ==================================================

def parse_weather_query_with_llm(query: str) -> dict:
    """
    Use LLM to parse user's weather query into structured JSON.
    """

    logger.info("LLM weather query parser started")
    logger.info(f"Parser input query: {query}")

    default_result = {
        "city": "",
        "query_type": "current_weather",
        "forecast_days": 1,
        "response_format": "text"
    }

    if client is None:
        logger.warning("LLM weather parser skipped because OPENAI_API_KEY is missing")
        return default_result

    prompt = f"""
You are a weather query parser for Young60, an assistant for senior citizens.

Convert the user's weather question into JSON only.

Allowed query_type values:
- current_weather
- general_forecast
- rain_forecast
- snow_forecast
- heat_forecast
- storm_forecast

Allowed response_format values:
- text
- table

Rules:
- Extract city/location from the user query.
- If user asks weather now/today/current, query_type should be current_weather.
- If user asks next N days, set forecast_days to N.
- Maximum forecast_days should be 16.
- If user asks rain/chance of rain, query_type should be rain_forecast.
- If user asks snow, query_type should be snow_forecast.
- If user asks heat/hot/temperature, query_type should be heat_forecast.
- If user asks storm/thunder/wind, query_type should be storm_forecast.
- If user asks table, response_format should be table.
- If user asks bullets/list/points/WhatsApp note, response_format should be text.
- If no city is found, city should be empty string.
- Return valid JSON only. No markdown. No explanation.

Examples:

User: Weather in Jaipur
JSON:
{{"city":"Jaipur","query_type":"current_weather","forecast_days":1,"response_format":"text"}}

User: Weather in Los Angeles for next 10 days
JSON:
{{"city":"Los Angeles","query_type":"general_forecast","forecast_days":10,"response_format":"text"}}

User: Will it rain in Delhi tomorrow?
JSON:
{{"city":"Delhi","query_type":"rain_forecast","forecast_days":2,"response_format":"text"}}

User: Show Mumbai weather in table format
JSON:
{{"city":"Mumbai","query_type":"current_weather","forecast_days":1,"response_format":"table"}}

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
        logger.info(f"Raw weather parser response: {raw_text}")

        parsed = safe_json_loads(raw_text)

        city = clean_text(parsed.get("city", ""))
        query_type = clean_text(parsed.get("query_type", "current_weather")).lower()
        response_format = clean_text(parsed.get("response_format", "text")).lower()

        try:
            forecast_days = int(parsed.get("forecast_days", 1))
        except Exception:
            forecast_days = 1

        if forecast_days < 1:
            forecast_days = 1

        if forecast_days > 16:
            forecast_days = 16

        allowed_query_types = {
            "current_weather",
            "general_forecast",
            "rain_forecast",
            "snow_forecast",
            "heat_forecast",
            "storm_forecast"
        }

        if query_type not in allowed_query_types:
            query_type = "current_weather"

        if response_format not in {"text", "table"}:
            response_format = "text"

        result = {
            "city": city,
            "query_type": query_type,
            "forecast_days": forecast_days,
            "response_format": response_format
        }

        logger.info(f"Parsed weather query: {result}")

        return result

    except Exception as error:
        logger.error(f"LLM weather parser failed: {error}")
        return default_result

def parse_weather_query_by_rules(query: str) -> dict:
    """
    Rule-based fallback parser for weather query.
    """

    logger.info("Rule-based weather query parser started")

    q = query.lower().strip()

    result = {
        "city": "",
        "query_type": "current_weather",
        "forecast_days": 1,
        "response_format": "text"
    }

    # -----------------------------
    # Response format
    # -----------------------------

    if any(word in q for word in ["table", "table format"]):
        result["response_format"] = "table"

    # -----------------------------
    # Forecast days
    # -----------------------------

    day_match = re.search(r"next\s+(\d+)\s+days?", q)

    if day_match:
        try:
            result["forecast_days"] = int(day_match.group(1))
            result["query_type"] = "general_forecast"
        except Exception:
            result["forecast_days"] = 1

    elif "tomorrow" in q:
        result["forecast_days"] = 2
        result["query_type"] = "general_forecast"

    elif any(word in q for word in ["next few days", "coming days"]):
        result["forecast_days"] = 3
        result["query_type"] = "general_forecast"

    # -----------------------------
    # Query type
    # -----------------------------

    if "rain" in q:
        result["query_type"] = "rain_forecast"

    if "snow" in q or "snowfall" in q:
        if "ignore snow" not in q and "ignore snowfall" not in q and "not required" not in q:
            result["query_type"] = "snow_forecast"

    if any(word in q for word in ["hot", "heat", "temperature"]):
        result["query_type"] = "heat_forecast"

    if any(word in q for word in ["storm", "thunder", "wind"]):
        result["query_type"] = "storm_forecast"

    # -----------------------------
    # City extraction
    # -----------------------------

    patterns = [
        r"weather\s+in\s+(.+)",
        r"forecast\s+in\s+(.+)",
        r"rain\s+in\s+(.+)",
        r"snow\s+in\s+(.+)",
        r"temperature\s+in\s+(.+)",
        r"weather\s+tomorrow\s+in\s+(.+)",
        r"weather\s+today\s+in\s+(.+)",
        r"(.+?)\s+weather\s+next\s+\d+\s+days?",
        r"(.+?)\s+weather\s+tomorrow",
        r"(.+?)\s+weather\s+today",
        r"in\s+(.+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)

        if match:
            city = clean_weather_location_text(match.group(1))

            if city:
                result["city"] = city
                break

    if result["forecast_days"] > 16:
        result["forecast_days"] = 16

    logger.info(f"Rule parser result: {result}")

    return result

def parse_weather_query(query: str) -> dict:
    """
    Main weather query parser.
    Prefer LLM city if available.
    Use rule parser only as fallback or for format/day hints.
    """

    llm_result = parse_weather_query_with_llm(query)
    rule_result = parse_weather_query_by_rules(query)

    final_result = llm_result.copy()

    # Important:
    # Do not let rule parser overwrite a clean LLM city.
    if not final_result.get("city") and rule_result.get("city"):
        final_result["city"] = rule_result["city"]

    # Use table hint from rules
    if rule_result.get("response_format") == "table":
        final_result["response_format"] = "table"

    # Use forecast_days from rule if LLM missed it
    if final_result.get("forecast_days", 1) == 1 and rule_result.get("forecast_days", 1) > 1:
        final_result["forecast_days"] = rule_result["forecast_days"]

    # Use query_type from rule only if LLM stayed current_weather
    if (
        final_result.get("query_type") == "current_weather"
        and rule_result.get("query_type") != "current_weather"
    ):
        final_result["query_type"] = rule_result["query_type"]

    # Final cleanup
    final_result["city"] = clean_weather_location_text(final_result.get("city", ""))

    if not final_result.get("city"):
        final_result["city"] = ""

    logger.info(f"Final parsed weather query: {final_result}")

    return final_result

# ==================================================
# Geocoding API
# ==================================================

def geocode_location(location: str):
    """
    Convert location name into latitude and longitude using Open-Meteo geocoding API.
    """

    logger.info(f"Geocoding started for location: {location}")

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        logger.info(f"Geocoding API status: {response.status_code}")

        if response.status_code != 200:
            return None

        data = response.json()
        results = data.get("results", [])

        if not results:
            logger.warning(f"No geocoding result found for: {location}")
            return None

        place = results[0]

        geocoded_location = {
            "name": place.get("name", location),
            "country": place.get("country", ""),
            "admin1": place.get("admin1", ""),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
            "timezone": place.get("timezone", "auto")
        }

        logger.info(f"Geocoding result: {geocoded_location}")

        return geocoded_location

    except requests.RequestException as error:
        logger.error(f"Geocoding API request failed: {error}")
        return None

    except Exception as error:
        logger.error(f"Unexpected geocoding error: {error}")
        return None


# ==================================================
# Weather API
# ==================================================

def call_open_meteo(params: dict, api_name: str):
    """
    Call Open-Meteo with retry for temporary failures.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    retry_status_codes = {502, 503, 504}

    for attempt in range(1, 4):
        try:
            logger.info(f"{api_name} attempt {attempt}/3")

            response = requests.get(url, params=params, timeout=15)

            logger.info(f"{api_name} status: {response.status_code}")

            if response.status_code == 200:
                return response.json()

            if response.status_code in retry_status_codes:
                wait_seconds = attempt * 2
                logger.warning(
                    f"{api_name} temporary issue {response.status_code}. Retrying after {wait_seconds} seconds."
                )
                time.sleep(wait_seconds)
                continue

            logger.warning(f"{api_name} non-retryable status: {response.status_code}")
            return None

        except requests.RequestException as error:
            wait_seconds = attempt * 2
            logger.error(
                f"{api_name} request failed: {error}. Retrying after {wait_seconds} seconds."
            )
            time.sleep(wait_seconds)

        except Exception as error:
            logger.error(f"{api_name} unexpected error: {error}")
            return None

    logger.warning(f"{api_name} failed after retries")
    return None


def fetch_current_weather(latitude: float, longitude: float, timezone: str = "auto"):
    """
    Fetch current weather from Open-Meteo.
    """

    logger.info(f"Current weather API call started for lat={latitude}, lon={longitude}")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "rain,"
            "showers,"
            "snowfall,"
            "weather_code,"
            "wind_speed_10m,"
            "wind_gusts_10m"
        ),
        "timezone": timezone
    }

    data = call_open_meteo(params=params, api_name="Current weather API")

    if data is None or "current" not in data:
        logger.warning("Current weather data not available")
        return None

    return data


def fetch_daily_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    forecast_days: int = 7
):
    """
    Fetch daily forecast covering rain, snow, heat, storm, wind.
    """

    forecast_days = safe_int(forecast_days, default_value=7, minimum=1, maximum=16)

    logger.info(
        f"Daily forecast API call started for lat={latitude}, lon={longitude}, days={forecast_days}"
    )

    primary_daily_variables = (
        "weather_code,"
        "temperature_2m_max,"
        "temperature_2m_min,"
        "apparent_temperature_max,"
        "apparent_temperature_min,"
        "precipitation_probability_max,"
        "precipitation_sum,"
        "rain_sum,"
        "showers_sum,"
        "snowfall_sum,"
        "wind_speed_10m_max,"
        "wind_gusts_10m_max"
    )

    fallback_daily_variables = (
        "weather_code,"
        "temperature_2m_max,"
        "temperature_2m_min,"
        "precipitation_sum,"
        "rain_sum,"
        "snowfall_sum,"
        "wind_speed_10m_max"
    )

    request_options = [
        {
            "name": "primary_daily_forecast",
            "daily": primary_daily_variables
        },
        {
            "name": "fallback_daily_forecast",
            "daily": fallback_daily_variables
        }
    ]

    for option in request_options:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": option["daily"],
            "forecast_days": forecast_days,
            "timezone": timezone
        }

        data = call_open_meteo(
            params=params,
            api_name=f"Daily forecast API - {option['name']}"
        )

        if data is not None and "daily" in data:
            data["young60_forecast_source"] = option["name"]
            return data

    logger.warning("Daily forecast data not available")
    return None


# ==================================================
# Weather Code Formatter
# ==================================================

def weather_code_to_text(weather_code) -> str:
    """
    Convert Open-Meteo weather code into simple text.
    """

    weather_map = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Cloudy",
        45: "Foggy",
        48: "Foggy",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        56: "Freezing drizzle",
        57: "Freezing drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        66: "Freezing rain",
        67: "Freezing rain",
        71: "Light snowfall",
        73: "Snowfall",
        75: "Heavy snowfall",
        77: "Snow grains",
        80: "Light rain showers",
        81: "Rain showers",
        82: "Heavy rain showers",
        85: "Light snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail",
    }

    try:
        return weather_map.get(int(weather_code), "Weather condition unavailable")
    except Exception:
        return "Weather condition unavailable"


# ==================================================
# Senior Advice
# ==================================================

def get_current_weather_advice(current: dict) -> str:
    """
    Senior-friendly advice for current weather.
    """

    advice = []

    try:
        temp = float(current.get("temperature_2m", 0))
        feels_like = float(current.get("apparent_temperature", temp))
        humidity = float(current.get("relative_humidity_2m", 0))
        rain = float(current.get("rain", 0))
        snowfall = float(current.get("snowfall", 0))
        wind = float(current.get("wind_speed_10m", 0))
        gusts = float(current.get("wind_gusts_10m", 0))
        weather_text = weather_code_to_text(current.get("weather_code", "")).lower()
    except Exception:
        return "Please check the weather before going outside."

    if temp >= 35 or feels_like >= 38:
        advice.append("It may feel very hot. Please drink water and avoid going outside in the afternoon.")

    if temp <= 10:
        advice.append("It may be cold. Please wear warm clothes before going outside.")

    if humidity >= 75 and temp >= 30:
        advice.append("Humidity is high, so it may feel uncomfortable. Please stay hydrated.")

    if rain > 0 or "rain" in weather_text or "drizzle" in weather_text:
        advice.append("Rain is possible. Please carry an umbrella and avoid slippery areas.")

    if snowfall > 0 or "snow" in weather_text:
        advice.append("Snow is possible. Please wear warm shoes and avoid slippery paths.")

    if "thunderstorm" in weather_text:
        advice.append("Thunderstorm is possible. It is safer to stay indoors.")

    if wind >= 35 or gusts >= 50:
        advice.append("Wind may be strong. Please avoid walking near trees, poles, or loose objects.")

    if not advice:
        advice.append("Weather looks manageable. Please still take normal precautions before going outside.")

    return " ".join(advice)


def get_forecast_advice(
    query_type: str,
    highest_rain_chance: float,
    highest_rain_day: str,
    highest_snowfall: float,
    highest_snow_day: str,
    highest_feels_like: float,
    hottest_day: str,
    highest_wind_gust: float,
    windiest_day: str,
    storm_days: list
) -> str:
    """
    Senior-friendly advice for forecast.
    """

    advice = []

    if query_type == "rain_forecast" or highest_rain_chance > 0:
        if highest_rain_chance >= 70:
            advice.append(f"Rain chances look high around {highest_rain_day}. Please carry an umbrella and avoid slippery areas.")
        elif highest_rain_chance >= 40:
            advice.append(f"There is a moderate chance of rain around {highest_rain_day}. Carrying an umbrella would be safer.")

    if query_type == "snow_forecast" or highest_snowfall > 0:
        if highest_snowfall > 0:
            advice.append(f"Snow is possible around {highest_snow_day}. Please wear warm clothes and avoid slippery paths.")
        else:
            advice.append("Snow does not look likely in the available forecast, but please check again before travel.")

    if query_type == "heat_forecast" or highest_feels_like >= 35:
        if highest_feels_like >= 40:
            advice.append(f"Heat may be high around {hottest_day}. Please avoid afternoon outdoor travel and drink water.")
        elif highest_feels_like >= 35:
            advice.append(f"It may feel hot around {hottest_day}. Please stay hydrated.")

    if query_type == "storm_forecast" or storm_days or highest_wind_gust >= 50:
        if storm_days:
            advice.append(f"Thunderstorm conditions may occur around {', '.join(storm_days[:3])}. It is safer to stay indoors during storms.")
        elif highest_wind_gust >= 50:
            advice.append(f"Strong wind gusts may occur around {windiest_day}. Please avoid open areas and loose objects.")

    if not advice:
        advice.append("Forecast looks manageable, but please check again before going outside or travelling.")

    return " ".join(advice)


# ==================================================
# Response Formatters
# ==================================================

def build_display_location(location_info: dict) -> str:
    """
    Build readable location name.
    """

    location_name = location_info.get("name", "")
    admin1 = location_info.get("admin1", "")
    country = location_info.get("country", "")

    parts = [location_name, admin1, country]

    return ", ".join([part for part in parts if part])


def format_current_weather_response(
    location_info: dict,
    weather_data: dict,
    location_source: str
) -> str:
    """
    Format current weather response for senior citizens.
    """

    current = weather_data.get("current", {})
    units = weather_data.get("current_units", {})

    weather_text = weather_code_to_text(current.get("weather_code"))

    display_location = build_display_location(location_info)

    temperature = current.get("temperature_2m", "N/A")
    feels_like = current.get("apparent_temperature", "N/A")
    humidity = current.get("relative_humidity_2m", "N/A")
    precipitation = current.get("precipitation", "N/A")
    rain = current.get("rain", "N/A")
    snowfall = current.get("snowfall", "N/A")
    wind_speed = current.get("wind_speed_10m", "N/A")
    wind_gusts = current.get("wind_gusts_10m", "N/A")

    temp_unit = units.get("temperature_2m", "°C")
    humidity_unit = units.get("relative_humidity_2m", "%")
    precipitation_unit = units.get("precipitation", "mm")
    wind_unit = units.get("wind_speed_10m", "km/h")

    advice = get_current_weather_advice(current)

    location_note = ""

    if location_source == "default":
        location_note = f"""
Note:
No location was clearly mentioned, so I used default location: {DEFAULT_WEATHER_LOCATION}
"""

    return f"""
🌤️ Weather for {display_location}

Condition:
{weather_text}

Temperature:
{temperature}{temp_unit}

Feels Like:
{feels_like}{temp_unit}

Humidity:
{humidity}{humidity_unit}

Precipitation:
{precipitation}{precipitation_unit}

Rain:
{rain}{precipitation_unit}

Snowfall:
{snowfall}{precipitation_unit}

Wind:
{wind_speed} {wind_unit}

Wind Gusts:
{wind_gusts} {wind_unit}

Advice for Seniors:
{advice}
{location_note}
Source:
Open-Meteo public weather API
"""


def format_forecast_response(
    location_info: dict,
    forecast_data: dict,
    query_type: str,
    forecast_days: int,
    response_format: str,
    original_query: str
) -> str:
    """
    Format multi-day weather forecast.

    Supports:
    - rain
    - snow
    - heat
    - storm/wind
    - general forecast

    Final presentation is passed to LLM so user can ask:
    - bullets
    - table
    - short summary
    - WhatsApp note
    - Hindi/Hinglish
    """

    daily = forecast_data.get("daily", {})
    units = forecast_data.get("daily_units", {})
    forecast_source = forecast_data.get("young60_forecast_source", "")

    dates = daily.get("time", [])
    weather_codes = daily.get("weather_code", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    apparent_max = daily.get("apparent_temperature_max", [])
    apparent_min = daily.get("apparent_temperature_min", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    precipitation_sum = daily.get("precipitation_sum", [])
    rain_sum = daily.get("rain_sum", [])
    snowfall_sum = daily.get("snowfall_sum", [])
    wind_speed_max = daily.get("wind_speed_10m_max", [])
    wind_gusts_max = daily.get("wind_gusts_10m_max", [])

    temp_unit = units.get("temperature_2m_max", "°C")
    apparent_unit = units.get("apparent_temperature_max", temp_unit)
    probability_unit = units.get("precipitation_probability_max", "%")
    precipitation_unit = units.get("precipitation_sum", "mm")
    wind_unit = units.get("wind_speed_10m_max", "km/h")

    display_location = build_display_location(location_info)

    table_rows = []
    text_rows = []

    highest_rain_chance = 0
    highest_rain_day = ""
    highest_snowfall = 0
    highest_snow_day = ""
    highest_feels_like = -999
    hottest_day = ""
    highest_wind_gust = 0
    windiest_day = ""
    storm_days = []

    for index, date in enumerate(dates):
        code = weather_codes[index] if index < len(weather_codes) else "N/A"
        condition = weather_code_to_text(code)

        max_temp = temp_max[index] if index < len(temp_max) else "N/A"
        min_temp = temp_min[index] if index < len(temp_min) else "N/A"

        feels_like_max = apparent_max[index] if index < len(apparent_max) else max_temp
        feels_like_min = apparent_min[index] if index < len(apparent_min) else min_temp

        rain_chance = rain_prob[index] if index < len(rain_prob) else "N/A"
        precipitation = precipitation_sum[index] if index < len(precipitation_sum) else "N/A"
        rain = rain_sum[index] if index < len(rain_sum) else "N/A"
        snow = snowfall_sum[index] if index < len(snowfall_sum) else "N/A"
        wind_speed = wind_speed_max[index] if index < len(wind_speed_max) else "N/A"
        wind_gust = wind_gusts_max[index] if index < len(wind_gusts_max) else "N/A"

        try:
            if rain_chance != "N/A" and float(rain_chance) > highest_rain_chance:
                highest_rain_chance = float(rain_chance)
                highest_rain_day = date
        except Exception:
            pass

        try:
            if snow != "N/A" and float(snow) > highest_snowfall:
                highest_snowfall = float(snow)
                highest_snow_day = date
        except Exception:
            pass

        try:
            if feels_like_max != "N/A" and float(feels_like_max) > highest_feels_like:
                highest_feels_like = float(feels_like_max)
                hottest_day = date
        except Exception:
            pass

        try:
            if wind_gust != "N/A" and float(wind_gust) > highest_wind_gust:
                highest_wind_gust = float(wind_gust)
                windiest_day = date
        except Exception:
            pass

        if "thunderstorm" in condition.lower():
            storm_days.append(date)

        table_rows.append(
            f"| {date} | {condition} | {min_temp}-{max_temp}{temp_unit} | "
            f"{feels_like_min}-{feels_like_max}{apparent_unit} | "
            f"{rain_chance}{probability_unit} | {rain}{precipitation_unit} | "
            f"{snow}{precipitation_unit} | {wind_speed}/{wind_gust} {wind_unit} |"
        )

        text_rows.append(
            f"{date}: {condition}, Temp {min_temp}-{max_temp}{temp_unit}, "
            f"Feels like {feels_like_min}-{feels_like_max}{apparent_unit}, "
            f"Rain chance {rain_chance}{probability_unit}, "
            f"Rain {rain}{precipitation_unit}, "
            f"Snow {snow}{precipitation_unit}, "
            f"Wind/Gust {wind_speed}/{wind_gust} {wind_unit}"
        )

    advice = get_forecast_advice(
        query_type=query_type,
        highest_rain_chance=highest_rain_chance,
        highest_rain_day=highest_rain_day,
        highest_snowfall=highest_snowfall,
        highest_snow_day=highest_snow_day,
        highest_feels_like=highest_feels_like,
        hottest_day=hottest_day,
        highest_wind_gust=highest_wind_gust,
        windiest_day=windiest_day,
        storm_days=storm_days
    )

    if query_type == "rain_forecast":
        title_icon = "🌧️"
        title_text = "Rain Forecast"
    elif query_type == "snow_forecast":
        title_icon = "❄️"
        title_text = "Snow Forecast"
    elif query_type == "heat_forecast":
        title_icon = "🔥"
        title_text = "Heat Forecast"
    elif query_type == "storm_forecast":
        title_icon = "⛈️"
        title_text = "Storm / Wind Forecast"
    else:
        title_icon = "🌤️"
        title_text = "Weather Forecast"

    probability_note = ""

    if not rain_prob:
        probability_note = """
Note:
Rain probability was not available from the API response, so rain amount/precipitation values are shown instead.
"""

    logger.info(f"Internal forecast path: {forecast_source}")

    table_header = """
| Date | Condition | Temperature | Feels Like | Rain Chance | Rain | Snow | Wind/Gust |
|---|---|---:|---:|---:|---:|---:|---:|
"""

    table_text = "\n".join(table_rows)
    text = "\n".join(text_rows)

    if response_format == "table":
        base_answer = f"""
{title_icon} {title_text} for {display_location}

Next {forecast_days} days:

{table_header}
{table_text}

Advice for Seniors:
{advice}
{probability_note}
Source:
Open-Meteo public weather forecast API
"""
    else:
        base_answer = f"""
{title_icon} {title_text} for {display_location}

Next {forecast_days} days:

{text}

Advice for Seniors:
{advice}
{probability_note}
Source:
Open-Meteo public weather forecast API
"""

    return format_weather_answer_with_llm(
        user_query=original_query,
        base_answer=base_answer
    )

def format_weather_answer_with_llm(
    user_query: str,
    base_answer: str
) -> str:
    """
    Use LLM only to format the final weather answer.

    Important:
    - Weather data already comes from API.
    - LLM must not invent weather values.
    - LLM only changes presentation/style.
    """

    if client is None:
        logger.warning("Final weather formatting skipped because OPENAI_API_KEY is missing")
        return base_answer

    logger.info("LLM final weather formatter started")

    prompt = f"""
You are formatting a weather response for Young60, an assistant for senior citizens.

Rules:
- Use only the weather information provided below.
- Do not invent any temperature, rain, snow, wind, storm, or date value.
- Follow the user's requested format if mentioned.
- If user asks bullets, use clear bullet points.
- If user asks table, use markdown table.
- If user asks short, keep it short.
- If user asks WhatsApp style, make it copy-paste friendly.
- Keep it simple for senior citizens.
- Do not show internal/debug fields.
- Keep the source line if present.
- Do not mention "Internal Forecast Path".

User query:
{user_query}

Weather data response to format:
{base_answer}

Now return the final formatted answer only.
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

        logger.info("LLM final weather formatter completed")

        return final_answer

    except Exception as error:
        logger.error(f"LLM final weather formatter failed: {error}")
        return base_answer


def format_weather_not_found_response(location: str) -> str:
    """
    Safe fallback when weather cannot be fetched.
    """

    return f"""
I could not fetch weather information for:

{location}

Please check the location spelling and try again.

Examples:
Weather in Rohini
Weather in Jaipur
Chance of rain in Miami next week
Will it snow in London tomorrow?
Storm forecast in Tokyo next few days
"""


# ==================================================
# Main Service Function
# ==================================================

def weather_help(query: str) -> str:
    """
    Main weather service for Young60.

    Flow:
    1. Parse user query using LLM + fallback rules
    2. Resolve location using geocoding API
    3. Fetch current weather or daily forecast
    4. Build factual weather answer from API data
    5. Use LLM only to format the final answer
    """

    logger.info("=" * 60)
    logger.info("Weather service started")
    logger.info(f"User query: {query}")
    logger.info("=" * 60)

    parsed_query = parse_weather_query(query)

    location = parsed_query.get("city", "")
    query_type = parsed_query.get("query_type", "current_weather")
    forecast_days = safe_int(parsed_query.get("forecast_days", 1), default_value=1)
    response_format = parsed_query.get("response_format", "text")

    location_source = "query"

    if not location:
        location = DEFAULT_WEATHER_LOCATION
        location_source = "default"

    logger.info(f"Final weather location: {location}")
    logger.info(f"Location source: {location_source}")
    logger.info(f"Weather query type: {query_type}")
    logger.info(f"Forecast days: {forecast_days}")
    logger.info(f"Response format: {response_format}")

    location_info = geocode_location(location)

    if location_info is None:
        logger.warning("Location geocoding failed")
        return format_weather_not_found_response(location)

    latitude = location_info.get("latitude")
    longitude = location_info.get("longitude")
    timezone = location_info.get("timezone", "auto")

    if latitude is None or longitude is None:
        logger.warning("Latitude/longitude missing after geocoding")
        return format_weather_not_found_response(location)

    # --------------------------------------------------
    # Current weather path
    # --------------------------------------------------

    if query_type == "current_weather" and forecast_days <= 1:
        weather_data = fetch_current_weather(
            latitude=latitude,
            longitude=longitude,
            timezone=timezone
        )

        if weather_data is None:
            logger.warning("Current weather data fetch failed")
            return format_weather_not_found_response(location)

        base_answer = format_current_weather_response(
            location_info=location_info,
            weather_data=weather_data,
            location_source=location_source
        )

        return format_weather_answer_with_llm(
            user_query=query,
            base_answer=base_answer
        )

    # --------------------------------------------------
    # Forecast path
    # --------------------------------------------------

    forecast_data = fetch_daily_forecast(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        forecast_days=forecast_days
    )

    if forecast_data is None:
        logger.warning("Forecast data fetch failed")
        return format_weather_not_found_response(location)

    return format_forecast_response(
        location_info=location_info,
        forecast_data=forecast_data,
        query_type=query_type,
        forecast_days=forecast_days,
        response_format=response_format,
        original_query=query
    )