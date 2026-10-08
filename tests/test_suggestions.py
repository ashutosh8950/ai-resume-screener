import json
import logging

import pytest
import requests

from resume_screener import llm
from resume_screener.llm import GeminiError, generate_gemini_suggestions
from resume_screener.suggestions import suggestions_for_analysis


API_KEY = "test-secret-gemini-key"
ANALYSIS = {
    "missing_technical_skills": ["django"],
    "missing_skills": ["django"],
    "partial_matches": [{
        "requirement": "Build Django services",
        "evidence": "Built a Python service for reporting.",
        "status_reason": "Capped to Partial: django not found in resume",
    }],
    "gaps": [{"requirement": "Deploy services on AWS", "evidence": ""}],
    "keyword_score": 0.2,
}
MODEL_SUGGESTIONS = {
    "missing_skills": [{"skill": "django", "tip": "Build a small demo."}],
    "partial_matches": [{
        "requirement": "Build Django services",
        "evidence": "Built a Python service for reporting.",
        "suggestion": "Add a Django project example.",
    }],
    "gaps": [{"requirement": "Deploy services on AWS", "suggestion": "Add a deployment example."}],
    "ats_tips": ["Use standard headings."],
}


class MockResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload


def _success_response(suggestions=MODEL_SUGGESTIONS):
    return MockResponse(200, {
        "candidates": [{"content": {"parts": [{"text": json.dumps(suggestions)}]}}]
    })


def test_gemini_success_sends_only_structured_summary(monkeypatch):
    calls = []

    def mock_post(url, **kwargs):
        calls.append((url, kwargs))
        return _success_response()

    monkeypatch.setattr(llm.requests, "post", mock_post)

    result = generate_gemini_suggestions(ANALYSIS, API_KEY)

    assert result == MODEL_SUGGESTIONS
    assert len(calls) == 1
    assert calls[0][1]["params"] == {"key": API_KEY}
    assert calls[0][1]["timeout"] == 20
    sent_text = calls[0][1]["json"]["contents"][0]["parts"][0]["text"]
    assert "Built a Python service for reporting." in sent_text
    assert "full resume" not in sent_text.lower()
    assert "job description" not in sent_text.lower()


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (429, "free-tier quota or rate limit"),
        (401, "rejected that API key"),
        (403, "rejected that API key"),
    ],
)
def test_gemini_quota_and_invalid_key_errors_fall_back(monkeypatch, status_code, message):
    monkeypatch.setattr(llm.requests, "post", lambda *args, **kwargs: MockResponse(status_code))

    result, notice, source = suggestions_for_analysis(ANALYSIS, API_KEY)

    assert source == "rule-based"
    assert "django" == result["missing_skills"][0]["skill"]
    assert message in notice
    assert API_KEY not in notice


def test_gemini_timeout_retries_once_then_falls_back(monkeypatch):
    attempts = []

    def timeout(*args, **kwargs):
        attempts.append(kwargs["timeout"])
        raise requests.Timeout("request timed out")

    monkeypatch.setattr(llm.requests, "post", timeout)

    result, notice, source = suggestions_for_analysis(ANALYSIS, API_KEY)

    assert attempts == [20, 20]
    assert source == "rule-based"
    assert result["partial_matches"]
    assert "timed out" in notice


def test_key_is_never_exposed_in_suggestion_results_or_logs(monkeypatch, caplog):
    monkeypatch.setattr(llm.requests, "post", lambda *args, **kwargs: MockResponse(401))

    with caplog.at_level(logging.DEBUG):
        result, notice, _ = suggestions_for_analysis(ANALYSIS, API_KEY)

    assert API_KEY not in json.dumps(result)
    assert API_KEY not in notice
    assert API_KEY not in caplog.text


def test_timeout_constant_is_bounded():
    assert llm.REQUEST_TIMEOUT_SECONDS == 20