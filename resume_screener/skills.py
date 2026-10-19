"""Categorized technical and professional skills loaded from JSON."""

import json
import re
from pathlib import Path


_SKILLS_PATH = Path(__file__).with_name("skills.json")
with _SKILLS_PATH.open(encoding="utf-8") as skills_file:
    SKILL_CATEGORIES = json.load(skills_file)

TECH_SKILLS = [
    entry["name"]
    for category in SKILL_CATEGORIES.values()
    for entry in category
]
TECHNICAL_SKILLS = [
    entry["name"]
    for category_name, category in SKILL_CATEGORIES.items()
    if category_name != "soft_skills"
    for entry in category
]
SKILL_ALIASES = {
    alias.lower(): entry["name"]
    for category in SKILL_CATEGORIES.values()
    for entry in category
    for alias in entry["aliases"]
}


def _contains_skill(text: str, skill: str) -> bool:
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(skill) + r"(?![A-Za-z0-9_])"
    if skill == "c":
        pattern = r"(?<![A-Za-z0-9_])c(?![A-Za-z0-9_+#])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_skills(text: str) -> list[str]:
    normalized = text.lower()
    found = {skill for skill in TECH_SKILLS if _contains_skill(normalized, skill)}
    for alias, canonical in SKILL_ALIASES.items():
        if _contains_skill(normalized, alias):
            found.add(canonical)
    github_check_text = re.sub(
        r"(?<![A-Za-z0-9_])github copilot(?![A-Za-z0-9_])",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    if not _contains_skill(github_check_text, "github"):
        found.discard("github")
    return [skill for skill in TECH_SKILLS if skill in found]


def extract_technical_skills(text: str) -> list[str]:
    found = set(extract_skills(text))
    return [skill for skill in TECHNICAL_SKILLS if skill in found]