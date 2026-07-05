import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

from core.service_logger import get_service_logger


# ==================================================
# Environment Setup
# ==================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==================================================
# Logger Setup
# ==================================================

logger = get_service_logger("general_service")


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


# ==================================================
# General Query Parser
# ==================================================

def parse_general_query_with_llm(query: str) -> dict:
    """
    Parse general user query into a structured task.

    This does not fetch live external data.
    It only classifies the type of help needed.
    """

    logger.info("LLM general query parser started")
    logger.info(f"Parser input query: {query}")

    default_result = {
        "task_type": "general_help",
        "response_format": "text",
        "language_style": "simple",
        "risk_level": "normal"
    }

    if client is None:
        logger.warning("LLM general parser skipped because OPENAI_API_KEY is missing")
        return default_result

    prompt = f"""
You are a query parser for Young60, an assistant for senior citizens.

Convert the user's query into JSON only.

Allowed task_type values:
- digital_help
- scam_safety
- banking_safety
- government_help
- message_drafting
- explanation
- step_by_step
- emergency_guidance
- general_help

Allowed response_format values:
- text
- table

Allowed language_style values:
- simple
- hindi
- hinglish
- whatsapp

Allowed risk_level values:
- normal
- sensitive
- urgent

Rules:
- If user asks how to use app/phone/WhatsApp/UPI/email/video call, use digital_help.
- If user asks fraud/scam/OTP/suspicious link/cyber safety, use scam_safety.
- If user asks bank/UPI/payment/card/password/PIN, use banking_safety.
- If user asks govt service/Aadhaar/PAN/passport/pension, use government_help.
- If user asks to write a message, SMS, WhatsApp note, email, use message_drafting.
- If user asks explain something simply, use explanation.
- If user asks steps/process/how to do, use step_by_step.
- If user mentions emergency, danger, accident, chest pain, breathing issue, use emergency_guidance and risk_level urgent.
- If user asks table/comparison, response_format should be table.
- If user asks Hindi, use hindi.
- If user asks Hinglish, use hinglish.
- If user asks WhatsApp style, use whatsapp.
- For financial, legal, medical, safety-sensitive topics, risk_level should be sensitive unless urgent.
- Return valid JSON only. No markdown. No explanation.

Examples:

User: How to use WhatsApp video call?
JSON:
{{"task_type":"digital_help","response_format":"text","language_style":"simple","risk_level":"normal"}}

User: OTP aaya hai bank se, kya karu?
JSON:
{{"task_type":"scam_safety","response_format":"text","language_style":"hinglish","risk_level":"sensitive"}}

User: Write a WhatsApp message to my son that I reached safely
JSON:
{{"task_type":"message_drafting","response_format":"text","language_style":"whatsapp","risk_level":"normal"}}

User: Explain UPI fraud in table
JSON:
{{"task_type":"scam_safety","response_format":"table","language_style":"simple","risk_level":"sensitive"}}

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
        logger.info(f"Raw general parser response: {raw_text}")

        parsed = safe_json_loads(raw_text)

        task_type = clean_text(parsed.get("task_type", "general_help")).lower()
        response_format = clean_text(parsed.get("response_format", "text")).lower()
        language_style = clean_text(parsed.get("language_style", "simple")).lower()
        risk_level = clean_text(parsed.get("risk_level", "normal")).lower()

        allowed_task_types = {
            "digital_help",
            "scam_safety",
            "banking_safety",
            "government_help",
            "message_drafting",
            "explanation",
            "step_by_step",
            "emergency_guidance",
            "general_help",
        }

        if task_type not in allowed_task_types:
            task_type = "general_help"

        if response_format not in {"text", "table"}:
            response_format = "text"

        if language_style not in {"simple", "hindi", "hinglish", "whatsapp"}:
            language_style = "simple"

        if risk_level not in {"normal", "sensitive", "urgent"}:
            risk_level = "normal"

        result = {
            "task_type": task_type,
            "response_format": response_format,
            "language_style": language_style,
            "risk_level": risk_level
        }

        logger.info(f"Parsed general query: {result}")

        return result

    except Exception as error:
        logger.error(f"LLM general parser failed: {error}")
        return default_result


def parse_general_query_by_rules(query: str) -> dict:
    """
    Fallback parser if LLM fails.
    """

    logger.info("Rule-based general query parser started")

    q = query.lower().strip()

    result = {
        "task_type": "general_help",
        "response_format": "text",
        "language_style": "simple",
        "risk_level": "normal"
    }

    if any(word in q for word in ["whatsapp", "video call", "phone", "mobile", "email", "app", "upi app"]):
        result["task_type"] = "digital_help"

    if any(word in q for word in ["fraud", "scam", "otp", "suspicious", "unknown link", "cyber"]):
        result["task_type"] = "scam_safety"
        result["risk_level"] = "sensitive"

    if any(word in q for word in ["bank", "upi", "card", "pin", "password", "payment"]):
        result["task_type"] = "banking_safety"
        result["risk_level"] = "sensitive"

    if any(word in q for word in ["aadhaar", "pan", "passport", "pension", "government"]):
        result["task_type"] = "government_help"

    if any(word in q for word in ["write", "draft", "message", "sms", "email"]):
        result["task_type"] = "message_drafting"

    if any(word in q for word in ["explain", "samjhao", "what is"]):
        result["task_type"] = "explanation"

    if any(word in q for word in ["how to", "steps", "process", "kaise"]):
        result["task_type"] = "step_by_step"

    if any(word in q for word in ["emergency", "accident", "danger", "chest pain", "breathing"]):
        result["task_type"] = "emergency_guidance"
        result["risk_level"] = "urgent"

    if any(word in q for word in ["table", "comparison"]):
        result["response_format"] = "table"

    if "hindi" in q:
        result["language_style"] = "hindi"

    if "hinglish" in q:
        result["language_style"] = "hinglish"

    if "whatsapp" in q:
        result["language_style"] = "whatsapp"

    logger.info(f"Rule parser result: {result}")

    return result


def parse_general_query(query: str) -> dict:
    """
    Main general query parser.
    """

    llm_result = parse_general_query_with_llm(query)
    rule_result = parse_general_query_by_rules(query)

    final_result = llm_result.copy()

    if llm_result.get("task_type") == "general_help" and rule_result.get("task_type") != "general_help":
        final_result["task_type"] = rule_result["task_type"]

    if rule_result.get("response_format") == "table":
        final_result["response_format"] = "table"

    if rule_result.get("language_style") != "simple":
        final_result["language_style"] = rule_result["language_style"]

    if rule_result.get("risk_level") in {"sensitive", "urgent"}:
        final_result["risk_level"] = rule_result["risk_level"]

    logger.info(f"Final parsed general query: {final_result}")

    return final_result


# ==================================================
# Answer Builder
# ==================================================

def build_system_guidance(parsed_query: dict) -> str:
    """
    Build safe system guidance for the final LLM answer.
    """

    task_type = parsed_query.get("task_type", "general_help")
    response_format = parsed_query.get("response_format", "text")
    language_style = parsed_query.get("language_style", "simple")
    risk_level = parsed_query.get("risk_level", "normal")

    guidance = """
You are Young60, a simple assistant for senior citizens.

Core rules:
- Be clear, calm, and practical.
- Use short sentences.
- Prefer step-by-step instructions.
- Avoid jargon.
- Do not pretend to have live information.
- Do not claim you completed any real-world action.
- Do not ask for passwords, OTPs, card PINs, full bank details, or sensitive documents.
"""

    if response_format == "table":
        guidance += """
Format:
- Use a clean markdown table.
- Add a short note below the table.
"""

    if language_style == "hindi":
        guidance += """
Language:
- Respond in simple Hindi.
"""

    elif language_style == "hinglish":
        guidance += """
Language:
- Respond in simple Hinglish.
"""

    elif language_style == "whatsapp":
        guidance += """
Language:
- Make it copy-paste friendly like a WhatsApp message.
"""

    if task_type == "digital_help":
        guidance += """
Task:
- Explain digital steps slowly.
- Mention what the user should tap/click.
- Include a caution if money, OTP, or passwords are involved.
"""

    elif task_type == "scam_safety":
        guidance += """
Task:
- Help user stay safe from scam/fraud.
- Tell user not to share OTP, PIN, password, or screen-sharing access.
- Tell user not to click suspicious links.
- Suggest contacting bank/cyber helpline if money is involved.
"""

    elif task_type == "banking_safety":
        guidance += """
Task:
- Give only general safety guidance.
- Do not provide financial advice.
- Do not ask for or process sensitive banking details.
- Remind user never to share OTP, PIN, CVV, password, or UPI PIN.
"""

    elif task_type == "government_help":
        guidance += """
Task:
- Give general guidance only.
- Do not claim current government rules unless provided by user.
- Suggest checking official website or visiting a trusted service center for final confirmation.
"""

    elif task_type == "message_drafting":
        guidance += """
Task:
- Draft a clear message.
- Keep it polite and short.
"""

    elif task_type == "emergency_guidance":
        guidance += """
Task:
- If there is immediate danger, tell user to contact local emergency services now.
- Do not give detailed medical treatment.
- Keep response urgent and simple.
"""

    if risk_level == "urgent":
        guidance += """
Urgency:
- Start with emergency guidance.
- Tell user to contact local emergency services immediately if there is danger.
"""

    elif risk_level == "sensitive":
        guidance += """
Safety:
- Add a clear safety note.
"""

    return guidance


def generate_general_answer_with_llm(
    user_query: str,
    parsed_query: dict
) -> str:
    """
    Generate final general answer using LLM.
    """

    if client is None:
        logger.warning("General LLM answer skipped because OPENAI_API_KEY is missing")

        return """
I can help with this, but the AI service is not configured right now.

Please check OPENAI_API_KEY in your .env file.
"""

    logger.info("LLM general answer generation started")

    system_guidance = build_system_guidance(parsed_query)

    prompt = f"""
User query:
{user_query}

Parsed query:
{json.dumps(parsed_query, ensure_ascii=False)}

Please answer the user.
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_guidance
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        answer = response.choices[0].message.content.strip()

        logger.info("LLM general answer generation completed")

        return answer

    except Exception as error:
        logger.error(f"LLM general answer generation failed: {error}")

        return """
Sorry, I could not prepare the answer right now.

Please try again in a moment.
"""


# ==================================================
# Main General Service
# ==================================================

def general_help(query: str) -> str:
    """
    Main general service for Young60.

    Flow:
    1. Parse general query using LLM + fallback rules
    2. Build safe guidance
    3. Generate senior-friendly answer
    """

    logger.info("=" * 60)
    logger.info("General service started")
    logger.info(f"User query: {query}")
    logger.info("=" * 60)

    parsed_query = parse_general_query(query)

    logger.info(f"General task type: {parsed_query.get('task_type')}")
    logger.info(f"Response format: {parsed_query.get('response_format')}")
    logger.info(f"Language style: {parsed_query.get('language_style')}")
    logger.info(f"Risk level: {parsed_query.get('risk_level')}")

    return generate_general_answer_with_llm(
        user_query=query,
        parsed_query=parsed_query
    )