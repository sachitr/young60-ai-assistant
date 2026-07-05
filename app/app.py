import sys
import os
import streamlit as st

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from core.router import route_query
from graph.young60_graph import young60_graph


# ==================================================
# Optional Browser Location Support
# ==================================================

try:
    from streamlit_js_eval import get_geolocation
except Exception:
    get_geolocation = None


# ==================================================
# Page Setup
# ==================================================

st.set_page_config(
    page_title="Young 60",
    page_icon="🧓",
    layout="centered"
)

st.title("Young 60")
st.caption("Your simple AI assistant for weather, medicine, nearby places, metro and general help.")


# ==================================================
# Session State
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Namaste 👋\n\n"
                "I am Young60. You can ask me about:\n\n"
                "- Weather\n"
                "- Medicine general awareness\n"
                "- Nearby places\n"
                "- Metro / public transport\n"
                "- General digital help\n\n"
                "Example: *Find hospital near Rohini*"
            )
        }
    ]

if "user_location" not in st.session_state:
    st.session_state.user_location = None

if "last_graph_trace" not in st.session_state:
    st.session_state.last_graph_trace = {}

if "session_summary" not in st.session_state:
    st.session_state.session_summary = ""

# ==================================================
# Sidebar Settings
# ==================================================

with st.sidebar:
    st.header("Settings")

    use_current_location = st.checkbox(
        label="Use my current location for nearby searches",
        value=False
    )

    if use_current_location:
        if get_geolocation is None:
            st.warning("Location feature is not installed.")
            st.caption("Run: pip install streamlit-js-eval")
        else:
            location_data = get_geolocation()

            if location_data and "coords" in location_data:
                latitude = location_data["coords"].get("latitude")
                longitude = location_data["coords"].get("longitude")

                if latitude is not None and longitude is not None:
                    st.session_state.user_location = {
                        "latitude": latitude,
                        "longitude": longitude
                    }

                    st.success("Current location detected.")
                else:
                    st.session_state.user_location = None
                    st.caption("Location permission allowed, but coordinates were not received.")
            else:
                st.session_state.user_location = None
                st.caption("Please allow location access in your browser.")
    else:
        st.session_state.user_location = None

    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Chat cleared. How can I help you now?\n\n"
                    "Example: *Weather in Jaipur*"
                )
            }
        ]
        
        st.session_state.last_graph_trace = {}
        st.session_state.session_summary = ""

        st.rerun()

    st.divider()

    with st.expander("Graph Debug", expanded=False):
        graph_trace = st.session_state.get("last_graph_trace", {})

        if graph_trace:
            st.json(graph_trace)
        else:
            st.caption("No graph trace yet.")

        st.divider()
        st.caption("Session Summary")

        session_summary = st.session_state.get("session_summary", "")

        if session_summary:
            st.write(session_summary)
        else:
            st.caption("No session summary yet.")


# ==================================================
# Display Chat History
# ==================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==================================================
# Chat Input
# ==================================================

user_query = st.chat_input(
    "Ask Young60..."
)


# ==================================================
# Process User Query
# ==================================================

if user_query:
    cleaned_query = user_query.strip()

    if cleaned_query:
        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": cleaned_query
            }
        )

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(cleaned_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            progress_area = st.empty()

            try:
                progress_area.info("Understanding your question...")

                with st.spinner("Young60 is working. Please wait..."):
                    graph_result = young60_graph.invoke(
                         {
                            "user_query": cleaned_query,
                            "messages": st.session_state.messages,
                            "session_summary": st.session_state.session_summary,
                            "user_location": st.session_state.user_location
                        }
                    )

                    response = graph_result.get(
                        "response",
                        "Sorry, I could not prepare a response."
                    )

                    st.session_state.last_graph_trace = graph_result.get("graph_trace", {})
                    st.session_state.session_summary = graph_result.get(
                        "updated_session_summary",
                        st.session_state.session_summary
                    )

                progress_area.empty()

                st.markdown(response)

                # Save assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            except Exception as error:
                progress_area.empty()

                error_message = (
                    "Sorry, something went wrong while processing your request.\n\n"
                    f"Technical detail: `{error}`"
                )

                st.error("Sorry, something went wrong while processing your request.")
                st.caption(f"Technical detail: {error}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )