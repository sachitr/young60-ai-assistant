from core.intent_engine import detect_intent


def route_query(query: str, user_location: dict = None) -> str:
    """
    Route the user query to the correct Young60 service.

    Flow:
    1. Detect intent using intent_engine
    2. Send query to the matching service
    3. Return service response
    """

    intent = detect_intent(query)

    print(f"[ROUTER] Query: {query}", flush=True)
    print(f"[ROUTER] Detected intent: {intent}", flush=True)

    if intent == "metro":
        from services.metro_service import metro_help
        return metro_help(query)

    if intent == "medicine":
        from services.medicine_service import medicine_help
        return medicine_help(query)

    if intent == "nearby":
        from services.nearby_service import nearby_help
        return nearby_help(query)

    if intent == "weather":
        from services.weather_service import weather_help
        return weather_help(query)

    from services.general_service import general_help
    return general_help(query)