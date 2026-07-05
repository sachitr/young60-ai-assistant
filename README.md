# Young60

Young60 is an AI-powered assistant for senior citizens.

It helps with:
- Weather queries
- Medicine general awareness
- Nearby places
- Metro / public transport queries
- General digital help
- Safety guidance for urgent or sensitive questions

## Architecture

User Chat
→ Streamlit Chat UI
→ LangGraph Orchestration
→ Memory / Query Rewrite / Safety Guard
→ Intent Detection
→ Service Nodes
→ Final Senior-Friendly Response

## Main Components

- `app/app.py` - Streamlit chatbot UI
- `graph/young60_graph.py` - LangGraph orchestration
- `graph/state.py` - Shared graph state
- `core/intent_engine.py` - Intent detection
- `services/` - Domain services
  - weather
  - medicine
  - nearby
  - metro
  - general help

## Setup

```bash
pip install -r requirements.txt

## GitHub Push

```bash
git init
git status
git add .
git commit -m "Initial Young60 LangGraph chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/young60-ai-assistant.git
git push -u origin main

```markdown
## Learning Outcomes

While building Young60, I learned how to design a multi-service AI assistant using LangGraph, Streamlit, external APIs, memory-aware query rewriting, and safety guardrails.

Detailed learning notes are available here:

[Learning Notes](docs/LEARNINGS.md)