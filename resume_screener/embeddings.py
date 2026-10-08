"""Lazy CPU embedding model, document segmentation, and calibrated matching."""

import re
import unicodedata

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from resume_screener.config import (
    EMBEDDING_MODEL,
    PARTIAL_MATCH_THRESHOLD,
    STRONG_MATCH_THRESHOLD,
)
from resume_screener.skills import extract_skills


_embedding_model = None
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BULLET_CHARS = "•▪◦▯□■●‣⁃◆◇"
_BULLET_PATTERN = re.compile(f"[{re.escape(_BULLET_CHARS)}]")
_LINE_BULLET_PATTERN = re.compile(r"^[ \t]*(?:[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_PATTERN = re.compile(r"[\w+#.]+", re.UNICODE)
_EXCLUDED_LINE = re.compile(
    r"\b(?:about (?:us|the company|our company)|who we are|company overview|our mission|our culture|"
    r"benefits?|salary|compensation|pay range|location|relocation|"
    r"working days?|working hours?|employment type|work schedule|holidays?|"
    r"work model|schedule|full[- ]time|part[- ]time|contract role|internship|"
    r"paid time off|vacation|health insurance|equal opportunity)\b",
    re.IGNORECASE,
)
_REQUIREMENT_CUE = re.compile(
    r"\b(?:responsibilit(?:y|ies)|require(?:ment|ments|d|s)|qualif(?:ication|ications|ied)|"
    r"experience|proficien(?:t|cy)|knowledge|ability|skills?|must have|should have|"
    r"develop|build|design|implement|maintain|deploy|manage|support|lead|create|"
    r"bachelor|degree|years?)\b",
    re.IGNORECASE,
)
_POSITIVE_HEADINGS = {
    "responsibilities", "key responsibilities", "core responsibilities", "what you will do", "what you'll do",
    "requirements", "job requirements", "minimum requirements", "qualifications",
    "required qualifications", "minimum qualifications", "preferred qualifications",
    "requirements and qualifications", "technical requirements", "skills", "technical skills",
    "required skills", "required experience", "what you'll bring", "what you will bring",
    "what we're looking for", "what we are looking for", "tools", "tools and technologies",
    "technologies", "programming languages", "required technical skills", "preferred skills", "desired skills",
    "must have", "nice to have", "your qualifications", "what we need",
}
_NEGATIVE_HEADINGS = {
    "about", "about us", "about the company", "about our company", "about the role",
    "company overview", "benefits", "perks", "compensation", "salary", "location",
    "work location", "working hours", "work schedule", "employment type", "equal opportunity",
}
_RESUME_HEADINGS = {
    "summary", "professional summary", "profile", "objective", "experience",
    "professional experience", "work experience", "employment history", "work history",
    "skills", "technical skills", "core competencies", "education", "projects",
    "personal projects", "certifications", "licenses", "awards", "publications",
    "volunteer experience", "languages", "references", "achievements",
    "career highlights", "key skills", "technical competencies", "selected projects",
    "relevant experience", "career objective", "professional development",
}
_SKILL_HEADINGS = {"skills", "technical skills", "core competencies", "technologies", "tools"}
_LINKEDIN_SPACE = re.compile(r"(?<=\w)(?=LinkedIn\b)")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\ufffd", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BULLET_PATTERN.sub(" ", text)
    text = _LINE_BULLET_PATTERN.sub("", text)
    text = _LINKEDIN_SPACE.sub(" ", text)
    return text


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def _heading_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w+#. ]", " ", line).strip().lower())


def _word_count(line: str) -> int:
    return len(_WORD_PATTERN.findall(line))


def _is_heading(line: str) -> tuple[bool, str]:
    stripped = clean_text(line).strip().rstrip(":")
    key = _heading_key(stripped)
    if key in _POSITIVE_HEADINGS:
        return True, "requirements"
    if key in _NEGATIVE_HEADINGS:
        return True, "excluded"
    if line.strip().endswith(":") or (stripped.isupper() and _word_count(stripped) <= 7):
        return True, "excluded"
    title_case = stripped.istitle() and _word_count(stripped) <= 6
    if title_case and re.match(r"^(about|benefits?|salary|location|compensation|employment)\b", key):
        return True, "excluded"
    return False, ""


def filter_job_description(job_description: str) -> tuple[list[str], list[dict]]:
    requirements = []
    ignored = []
    section = None
    for raw_line in job_description.splitlines():
        line = clean_text(raw_line).strip()
        if not line:
            continue
        is_heading, heading_kind = _is_heading(line)
        if is_heading:
            section = heading_kind
            ignored.append({"line": line, "reason": "section heading"})
            continue
        if _EXCLUDED_LINE.search(line):
            ignored.append({"line": line, "reason": "benefits, compensation, location, or schedule"})
            continue
        if _word_count(line) < 4:
            ignored.append({"line": line, "reason": "fewer than four words"})
            continue
        if section == "excluded":
            ignored.append({"line": line, "reason": "non-requirement section"})
            continue
        requirement_like = section == "requirements" or bool(_REQUIREMENT_CUE.search(line))
        requirement_like = requirement_like or bool(extract_skills(line))
        if not requirement_like:
            ignored.append({"line": line, "reason": "not requirement-focused"})
            continue
        parts = split_sentences(line)
        for part in parts:
            if _word_count(part) >= 4:
                requirements.append(part)
            else:
                ignored.append({"line": part, "reason": "sentence has fewer than four words"})
    return requirements, ignored


def _resume_heading(line: str) -> tuple[bool, str, bool]:
    stripped = clean_text(line).strip().rstrip(":")
    key = _heading_key(stripped)
    inline = re.match(r"^(technical skills|core competencies|skills)\s*:\s*(.+)$", stripped, re.IGNORECASE)
    if inline:
        return True, inline.group(1), True
    if key in _RESUME_HEADINGS:
        return True, stripped, key in _SKILL_HEADINGS
    if stripped.isupper() and _word_count(stripped) <= 6 and not extract_skills(stripped):
        return True, stripped, False
    return False, "", False


def segment_resume(resume: str) -> list[dict]:
    segments = []
    section = "Resume"
    skills_section = False
    for raw_line in resume.splitlines():
        is_bullet = bool(re.match(rf"^\s*(?:[{re.escape(_BULLET_CHARS)}]|[-*+]|\d+[.)])\s*", raw_line))
        is_heading, heading, inline_skills = _resume_heading(raw_line)
        if is_heading:
            section = heading
            skills_section = inline_skills or _heading_key(heading) in _SKILL_HEADINGS
            inline = re.match(r"^(?:technical skills|core competencies|skills)\s*:\s*(.+)$", raw_line, re.IGNORECASE)
            if inline:
                raw_line = inline.group(1)
                is_bullet = False
            else:
                continue
        text = clean_text(raw_line).strip()
        if not text or _word_count(text) < 2:
            if text and segments and not is_bullet:
                segments[-1]["text"] += " " + text
            continue
        unit = {"text": text, "section": section, "skills_only": skills_section}
        if (
            segments
            and not is_bullet
            and segments[-1]["section"] == section
            and not skills_section
            and (not re.search(r"[.!?;:]$", segments[-1]["text"]) or _word_count(text) <= 7)
        ):
            segments[-1]["text"] += " " + text
        else:
            segments.append(unit)
    return segments


def get_embedding_model():
    """Create the fastembed model once, on the first requested analysis."""
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding

        _embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL,
            providers=["CPUExecutionProvider"],
        )
    return _embedding_model


def rescale_similarity(cosine: float) -> float:
    if cosine <= PARTIAL_MATCH_THRESHOLD:
        return 0.0
    return float(np.clip(
        (cosine - PARTIAL_MATCH_THRESHOLD) / (1.0 - PARTIAL_MATCH_THRESHOLD),
        0.0,
        1.0,
    ))


def classify_similarity(cosine: float) -> str:
    if cosine >= STRONG_MATCH_THRESHOLD:
        return "Strong"
    if cosine >= PARTIAL_MATCH_THRESHOLD:
        return "Partial"
    return "Gap"


def match_requirements(
    resume: str,
    job_description: str,
    requirements: list[str] | None = None,
    ignored_lines: list[dict] | None = None,
) -> dict:
    if requirements is None:
        requirements, detected_ignored_lines = filter_job_description(job_description)
        ignored_lines = detected_ignored_lines
    ignored_lines = ignored_lines or []
    evidence = segment_resume(resume)
    if not requirements:
        return {
            "semantic_score": 0.0,
            "strong_matches": [],
            "partial_matches": [],
            "gaps": [],
            "ignored_lines": ignored_lines,
            "requirements": [],
            "raw_match_scores": [],
        }

    resume_skills = set(extract_skills(resume))
    embedded_evidence = []
    for requirement in requirements:
        requirement_skills = set(extract_skills(requirement))
        embedded_evidence.append([
            item for item in evidence
            if not item["skills_only"] or bool(requirement_skills)
        ])
    queries = [BGE_QUERY_PREFIX + requirement for requirement in requirements]
    passages = [item["text"] for group in embedded_evidence for item in group]
    if not passages:
        pairs = [
            {
                "requirement": requirement,
                "evidence": "",
                "section": "",
                "score": 0.0,
                "raw_score": 0.0,
                "status": "Gap",
                "missing_skills": sorted(set(extract_skills(requirement)) - resume_skills),
            }
            for requirement in requirements
        ]
        return {
            "semantic_score": 0.0,
            "strong_matches": [],
            "partial_matches": [],
            "gaps": pairs,
            "ignored_lines": ignored_lines,
            "requirements": requirements,
            "raw_match_scores": pairs,
        }

    model = get_embedding_model()
    vectors = np.asarray(list(model.embed(queries + passages)), dtype=np.float32)
    query_vectors = vectors[:len(queries)]
    passage_vectors = vectors[len(queries):]

    strong_matches = []
    partial_matches = []
    gaps = []
    raw_match_scores = []
    offset = 0
    scaled_scores = []
    for index, requirement in enumerate(requirements):
        items = embedded_evidence[index]
        requirement_skills = set(extract_skills(requirement))
        missing_skills = sorted(requirement_skills - resume_skills)
        if not items:
            pair = {
                "requirement": requirement,
                "evidence": "",
                "section": "",
                "score": 0.0,
                "raw_score": 0.0,
                "status": "Gap",
                "missing_skills": missing_skills,
            }
            gaps.append(pair)
            raw_match_scores.append(pair)
            scaled_scores.append(0.0)
            continue
        item_vectors = passage_vectors[offset:offset + len(items)]
        offset += len(items)
        row = cosine_similarity(query_vectors[index:index + 1], item_vectors)[0]
        best_index = int(np.argmax(row))
        raw_score = float(np.clip(row[best_index], 0.0, 1.0))
        scaled_score = rescale_similarity(raw_score)
        status = classify_similarity(raw_score)
        if missing_skills:
            if status == "Strong":
                status = "Partial"
            if not requirement_skills.intersection(resume_skills):
                status = "Gap"
        best_evidence = items[best_index]
        pair = {
            "requirement": requirement,
            "evidence": best_evidence["text"],
            "section": best_evidence["section"],
            "score": round(scaled_score, 4),
            "raw_score": round(raw_score, 4),
            "status": status,
            "missing_skills": missing_skills,
        }
        raw_match_scores.append(pair)
        scaled_scores.append(scaled_score)
        if status == "Strong":
            strong_matches.append(pair)
        elif status == "Partial":
            partial_matches.append(pair)
        else:
            gaps.append(pair)

    strong_matches.sort(key=lambda pair: pair["raw_score"], reverse=True)
    partial_matches.sort(key=lambda pair: pair["raw_score"], reverse=True)
    gaps.sort(key=lambda pair: pair["raw_score"])
    return {
        "semantic_score": float(np.mean(scaled_scores)),
        "strong_matches": strong_matches,
        "partial_matches": partial_matches,
        "gaps": gaps,
        "ignored_lines": ignored_lines,
        "requirements": requirements,
        "raw_match_scores": raw_match_scores,
    }