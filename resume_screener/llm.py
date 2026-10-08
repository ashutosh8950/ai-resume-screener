"""Optional Gemini suggestions using its HTTPS API without an SDK dependency."""

import json

import requests

from resume_screener.config import GEMINI_MODEL


REQUEST_TIMEOUT_SECONDS = 20
_MAX_ITEMS_PER_SECTION = 5
_SECTIONS = ("missing_skills", "partial_matches", "gaps", "ats_tips")
_SECTION_FIELDS = {
    "missing_skills": ("skill", "tip"),
    "partial_matches": ("requirement", "evidence", "suggestion"),
    "gaps": ("requirement", "suggestion"),
}


class GeminiError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def _analysis_summary(analysis: dict) -> dict:
    return {
        "missing_skills": list(dict.fromkeys(
            analysis.get("missing_technical_skills") or analysis.get("missing_skills", [])
        ))[:_MAX_ITEMS_PER_SECTION],
        "partial_requirements": [
            {
                "requirement": item.get("requirement", "")[:240],
                "evidence": item.get("evidence", "")[:240],
            }
            for item in analysis.get("partial_matches", [])[:_MAX_ITEMS_PER_SECTION]
        ],
        "gaps": [
            {"requirement": item.get("requirement", "")[:240]}
            for item in analysis.get("gaps", [])[:_MAX_ITEMS_PER_SECTION]
        ],
        "keyword_score": analysis.get("keyword_score", 0.0),
    }


def _validated_suggestions(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise GeminiError("Gemini returned an unusable response. Showing rule-based suggestions instead.")
    output = {}
    for section in _SECTIONS:
        items = payload.get(section, [])
        if not isinstance(items, list):
            raise GeminiError("Gemini returned an unusable response. Showing rule-based suggestions instead.")
        output[section] = items[:_MAX_ITEMS_PER_SECTION]
        for item in output[section]:
            if section == "ats_tips":
                if not isinstance(item, str):
                    raise GeminiError("Gemini returned an unusable response. Showing rule-based suggestions instead.")
            elif not isinstance(item, dict) or any(
                not isinstance(item.get(field), str)
                for field in _SECTION_FIELDS[section]
            ):
                raise GeminiError("Gemini returned an unusable response. Showing rule-based suggestions instead.")
    return output


def generate_gemini_suggestions(analysis: dict, api_key: str) -> dict:
    summary = _analysis_summary(analysis)
    prompt = (
        "Return only JSON with sections missing_skills, partial_matches, gaps, and ats_tips. "
        "Use missing_skills items {skill, tip}; partial_matches items {requirement, evidence, suggestion}; "
        "gaps items {requirement, suggestion}; and ats_tips strings. Give concise, truthful resume advice. "
        "Use only the supplied structured summary and never invent experience.\nSUMMARY:\n"
        + json.dumps(summary, ensure_ascii=True)
    )
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    request_body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }

    for attempt in range(2):
        try:
            response = requests.post(
                endpoint,
                params={"key": api_key},
                json=request_body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            if attempt == 0:
                continue
            raise GeminiError("Gemini timed out. Showing rule-based suggestions instead.") from None
        except requests.RequestException:
            if attempt == 0:
                continue
            raise GeminiError("Gemini is temporarily unavailable. Showing rule-based suggestions instead.") from None

        if response.status_code in (401, 403):
            raise GeminiError("Gemini rejected that API key. Check it or clear it to use rule-based suggestions.")
        if response.status_code == 429:
            raise GeminiError("Gemini's free-tier quota or rate limit was reached. Showing rule-based suggestions instead.")
        if response.status_code >= 500 and attempt == 0:
            continue
        if response.status_code < 200 or response.status_code >= 300:
            raise GeminiError("Gemini could not generate suggestions. Showing rule-based suggestions instead.")
        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return _validated_suggestions(json.loads(text))
        except GeminiError:
            raise
        except (ValueError, KeyError, IndexError, TypeError):
            raise GeminiError("Gemini returned an unusable response. Showing rule-based suggestions instead.") from None

    raise GeminiError("Gemini is temporarily unavailable. Showing rule-based suggestions instead.")