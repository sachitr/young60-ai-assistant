# Young60 - Learning Notes

## What I Built

Young60 is a chatbot-style AI assistant designed for senior citizens.

The app supports:

- Weather queries
- Medicine general awareness
- Nearby place search
- Metro / public transport queries
- General digital help
- Safety guard for urgent or sensitive situations

## Key Technical Learnings

### 1. LLM should not be the only source of truth

I learned that an LLM is very useful for understanding natural language and formatting responses, but it should not be trusted blindly for factual services.

For example:

- Weather data should come from a weather API
- Nearby places should come from map/OpenStreetMap data
- Medicine information should come from verified data sources where possible
- Metro routes should ideally come from a transport API

The LLM is best used for:

- Parsing user intent
- Extracting structured JSON
- Rewriting follow-up questions
- Formatting final senior-friendly answers

---

### 2. Service-based architecture is cleaner

Instead of putting everything in one file, I separated the app into services:

- `weather_service.py`
- `medicine_service.py`
- `nearby_service.py`
- `metro_service.py`
- `general_service.py`

This made the app easier to extend and debug.

---

### 3. LangGraph helps convert simple routing into agentic flow

Initially, the app used a simple router.

Later, I converted it into a LangGraph flow with nodes:

1. Build memory context
2. Rewrite follow-up query
3. Safety guard
4. Detect intent
5. Route to service node
6. Final response cleanup
7. Update session summary

This made the assistant more structured and production-style.

---

### 4. Memory makes the chatbot more natural

A one-shot app cannot understand follow-up questions.

With session memory and query rewriting, the chatbot can handle examples like:

User:

> Weather in Jaipur

Follow-up:

> What about tomorrow?

Internal rewritten query:

> Weather in Jaipur tomorrow

This makes the assistant feel more conversational.

---

### 5. Safety guard is important for senior citizen apps

Since Young60 is designed for senior citizens, I added a safety guard before normal routing.

It catches risky topics like:

- Chest pain
- Breathing difficulty
- Medicine overdose
- OTP sharing
- UPI PIN / CVV / password sharing

For such cases, the app gives direct safety guidance instead of routing normally.

---

### 6. Streamlit chat UI improves user experience

I moved from a simple text input to a chatbot interface using:

- `st.chat_message`
- `st.chat_input`
- `st.session_state.messages`

This made the app feel like a real assistant instead of a form-based tool.

---

### 7. Debug trace helps during development

I added a Graph Debug section in the sidebar to show:

- Original user query
- Rewritten query
- Intent
- Selected service node
- Safety status
- Session summary

This made debugging LangGraph much easier.

---

## Challenges Faced

### 1. Prompt JSON inside Python f-strings

When using JSON examples inside Python f-strings, braces must be escaped.

Wrong:

```python
{"city":"Jaipur"}
```

Correct:

```python
{{"city":"Jaipur"}}
```

This issue appeared in weather and nearby service prompts.

---

### 2. LLM output needs safe parsing

LLMs may return JSON inside markdown code blocks.

So I created helper functions like:

- `safe_json_loads`
- `clean_text`

These make parsing more reliable.

---

### 3. Nearby and metro services need more refinement

The nearby and metro services work as MVP, but they need better production-level data sources and validation.

This is a future improvement area.

---

## Current Architecture

```text
User
  ↓
Streamlit Chat UI
  ↓
LangGraph
  ↓
Memory Context
  ↓
Query Rewrite
  ↓
Safety Guard
  ↓
Intent Detection
  ↓
Conditional Service Node
  ↓
Final Response
```

---

## Future Improvements

- Add persistent memory with SQLite
- Improve nearby service accuracy
- Improve metro route quality
- Add voice input/output
- Add senior-friendly large button mode
- Add emergency contacts feature
- Add Hindi/Hinglish support across all services
- Add authentication and deployment
- Deploy on Streamlit Cloud or another platform

---

## Summary

This project helped me understand how to combine:

- LLMs
- APIs
- LangGraph
- Memory
- Safety checks
- Streamlit chatbot UI
- Service-based architecture

into one practical AI assistant application.