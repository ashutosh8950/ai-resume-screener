"""Build structured rule-based suggestions and optionally enhance them with Gemini."""

import json
import re
from pathlib import Path


_SUGGESTION_TIPS = json.loads(
    Path(__file__).with_name("suggestions.json").read_text(encoding="utf-8")
)


def build_rule_suggestions(analysis: dict) -> dict:
    missing_skills = list(dict.fromkeys(
        analysis.get("missing_technical_skills") or analysis.get("missing_skills", [])
    ))[:5]
    partial_matches = []
    partial_phrases = (
        "the relevant tool, your action, and the outcome",
        "a concise example naming the tool and result",
        "your contribution and measurable impact",
    )
    for index, match in enumerate(analysis.get("partial_matches", [])[:5]):
        requirement = match.get("requirement", "this requirement")
        missing_in_requirement = [
            skill for skill in missing_skills if skill.lower() in requirement.lower()
        ]
        if missing_in_requirement:
            addition = (
                f"a truthful project example using {', '.join(missing_in_requirement)}, "
                "including your contribution and outcome"
            )
        else:
            addition = partial_phrases[index % len(partial_phrases)]
        partial_matches.append({
            "requirement": requirement,
            "evidence": match.get("evidence", "")[:240],
            "suggestion": addition,
        })

    gaps = [
        {
            "requirement": match.get("requirement", "this requirement"),
            "suggestion": (
                "Add a truthful example showing the tool or skill used, the work you did, "
                "and its result."
            ),
        }
        for match in analysis.get("gaps", [])[:5]
    ]
    ats_tips = []
    if analysis.get("keyword_score", 1.0) < 0.5:
        ats_tips = [
            "Use standard headings such as Experience, Projects, Education, and Skills.",
            "Include relevant job-description keywords in context within truthful evidence, not as an isolated list.",
            "Quantify results with scale, time, reliability, users, or other meaningful measures.",
        ]

    return {
        "missing_skills": [
            {
                "skill": skill,
                "tip": _SUGGESTION_TIPS.get(
                    skill.lower(),
                    f"Build a small project using {skill}, then describe your contribution and a verifiable result.",
                ),
            }
            for skill in missing_skills
        ],
        "partial_matches": partial_matches,
        "gaps": gaps,
        "ats_tips": ats_tips,
    }


def suggestions_for_analysis(analysis: dict, api_key: str = "") -> tuple[dict, str, str]:
    rule_suggestions = build_rule_suggestions(analysis)
    if not api_key:
        return rule_suggestions, "", "rule-based"

    from resume_screener.llm import GeminiError, generate_gemini_suggestions

    try:
        return generate_gemini_suggestions(analysis, api_key), "", "gemini"
    except GeminiError as error:
        return rule_suggestions, error.user_message, "rule-based"