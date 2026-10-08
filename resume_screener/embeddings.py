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
from resume_screener.skills import (
    SKILL_CATEGORIES,
    extract_skills,
    extract_technical_skills,
)


_embedding_model = None
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_LINE_BULLET_CHARS = "•·▪➢-–—"
_BOX_CHARS = "□■▯▱▭▮▰▢◻◼◾◽▣▤▥▦▧▨▩⬜⬛"
_BULLET_CHARS = _LINE_BULLET_CHARS + _BOX_CHARS + "●‣⁃◆◇"
_LINE_BULLET_PATTERN = re.compile(rf"^[ \t]*[{re.escape(_LINE_BULLET_CHARS)}][ \t]*")
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
_OPTIONAL_HEADINGS = {
    "preferred", "preferred qualifications", "preferred skills", "nice to have",
    "preferred experience", "good to have", "optional", "bonus skills",
    "desired qualifications", "preferred nice to have",
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
_STRAY_INITIAL = re.compile(r"(?<!\S)([B-HJ-Z])\s+([a-z])")
_CAMEL_CASE_EXCEPTIONS = {
    "linkedin", "github", "javascript", "typescript", "pytorch", "fastapi",
    "opencv", "mongodb", "postgresql", "mysql", "gridfs",
}
for _skill_category in SKILL_CATEGORIES.values():
    for _skill_entry in _skill_category:
        for _skill_name in [_skill_entry["name"], *_skill_entry["aliases"]]:
            _compact_skill_name = re.sub(r"[^A-Za-z0-9+#]", "", _skill_name).lower()
            if _compact_skill_name:
                _CAMEL_CASE_EXCEPTIONS.add(_compact_skill_name)


def _split_camel_case(word: str) -> str:
    lowered = word.lower()
    protected_spans = []
    for exception in _CAMEL_CASE_EXCEPTIONS:
        if len(exception) < 3 and lowered != exception:
            continue
        start = lowered.find(exception)
        while start >= 0:
            protected_spans.append((start, start + len(exception)))
            start = lowered.find(exception, start + len(exception))

    output = []
    for index, char in enumerate(word):
        if index and word[index - 1].islower() and char.isupper():
            inside_exception = any(start < index < end for start, end in protected_spans)
            if not inside_exception:
                output.append(" ")
        output.append(char)
    return "".join(output)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned_chars = []
    for char in text:
        category = unicodedata.category(char)
        if char in "\n\t":
            cleaned_chars.append(char)
        elif char == "\ufffd" or char in _BOX_CHARS or category.startswith("C"):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(char)
    text = "".join(cleaned_chars)
    lines = []
    for line in text.splitlines():
        line = _LINE_BULLET_PATTERN.sub("", line)
        line = re.sub(r"\s*\|\s*", "\n", line)
        for segment in line.splitlines():
            if "://" not in segment:
                segment = re.sub(r"^([ \t]*[\w /&().-]{1,40}:)(?=\S)", r"\1 ", segment)
            segment = _STRAY_INITIAL.sub(r"\1\2", segment)
            words = re.split(r"(\s+)", segment)
            words = [
                word if not word or word.isspace() else _split_camel_case(word)
                for word in words
            ]
            lines.append("".join(words).strip())
    return "\n".join(line for line in lines if line)


def clean_text(text: str) -> str:
    """Backward-compatible alias for the shared text normalizer."""
    return normalize_text(text)


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def _heading_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w+#. ]", " ", line).strip().lower())


def _word_count(line: str) -> int:
    return len(_WORD_PATTERN.findall(line))


def _is_heading(line: str) -> tuple[bool, str, bool]:
    stripped = normalize_text(line).strip().rstrip(":")
    key = _heading_key(stripped)
    if key in _OPTIONAL_HEADINGS:
        return True, "requirements", True
    if key in _POSITIVE_HEADINGS:
        return True, "requirements", False
    if key in _NEGATIVE_HEADINGS:
        return True, "excluded", False
    if line.strip().endswith(":") or (stripped.isupper() and _word_count(stripped) <= 7):
        return True, "excluded", False
    title_case = stripped.istitle() and _word_count(stripped) <= 6
    if title_case and re.match(r"^(about|benefits?|salary|location|compensation|employment)\b", key):
        return True, "excluded", False
    return False, "", False


def filter_job_description(job_description: str) -> tuple[list[str], list[dict]]:
    requirements = []
    ignored = []
    section = None
    optional_section = False
    for raw_line in job_description.splitlines():
        line = normalize_text(raw_line).strip()
        if not line:
            continue
        is_heading, heading_kind, heading_optional = _is_heading(line)
        if is_heading:
            section = heading_kind
            optional_section = heading_optional
            ignored.append({"line": line, "reason": "section heading"})
            continue
        line_skills = extract_skills(line)
        if _EXCLUDED_LINE.search(line):
            ignored.append({"line": line, "reason": "benefits, compensation, location, or schedule"})
            continue
        if section == "excluded":
            ignored.append({"line": line, "reason": "non-requirement section"})
            continue
        if _word_count(line) < 4 and not line_skills and not optional_section:
            ignored.append({"line": line, "reason": "fewer than four words"})
            continue
        requirement_like = section == "requirements" or bool(_REQUIREMENT_CUE.search(line))
        requirement_like = requirement_like or bool(line_skills) or optional_section
        if not requirement_like:
            ignored.append({"line": line, "reason": "not requirement-focused"})
            continue
        parts = split_sentences(line)
        for part in parts:
            part_skills = extract_skills(part)
            if _word_count(part) >= 4 or part_skills or optional_section:
                requirements.append({"text": part, "optional": optional_section})
            else:
                ignored.append({"line": part, "reason": "sentence has fewer than four words"})
    return requirements, ignored


def _resume_heading(line: str) -> tuple[bool, str, bool]:
    stripped = normalize_text(line).strip().rstrip(":")
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
        text = normalize_text(raw_line).strip()
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


def coverage_credit(cosine: float, status: str) -> float:
    if status == "Gap":
        return 0.0
    if status == "Partial":
        progress = (cosine - PARTIAL_MATCH_THRESHOLD) / (
            STRONG_MATCH_THRESHOLD - PARTIAL_MATCH_THRESHOLD
        )
        return float(0.40 + 0.30 * np.clip(progress, 0.0, 1.0))
    progress = min(1.0, max(0.0, (cosine - STRONG_MATCH_THRESHOLD) / 0.15))
    return 0.80 + 0.20 * progress


def classify_similarity(cosine: float) -> str:
    if cosine >= STRONG_MATCH_THRESHOLD:
        return "Strong"
    if cosine >= PARTIAL_MATCH_THRESHOLD:
        return "Partial"
    return "Gap"


def match_requirements(
    resume: str,
    job_description: str,
    requirements: list[str | dict] | None = None,
    ignored_lines: list[dict] | None = None,
) -> dict:
    if requirements is None:
        requirements, detected_ignored_lines = filter_job_description(job_description)
        ignored_lines = detected_ignored_lines
    ignored_lines = ignored_lines or []
    requirement_records = [
        {"text": item, "optional": False} if isinstance(item, str) else item
        for item in requirements
    ]
    requirement_texts = [normalize_text(item["text"]) for item in requirement_records]
    evidence = segment_resume(resume)
    if not requirement_records:
        return {
            "semantic_score": 0.0,
            "strong_matches": [],
            "partial_matches": [],
            "gaps": [],
            "ignored_lines": ignored_lines,
            "requirements": [],
            "raw_match_scores": [],
        }

    resume_skills = set(extract_technical_skills(resume))
    embedded_evidence = []
    for requirement in requirement_texts:
        requirement_skills = set(extract_technical_skills(requirement))
        embedded_evidence.append([
            item for item in evidence
            if not item["skills_only"] or bool(requirement_skills)
        ])
    queries = [BGE_QUERY_PREFIX + requirement for requirement in requirement_texts]
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
                "optional": requirement_records[index].get("optional", False),
                "missing_skills": sorted(set(extract_technical_skills(requirement)) - resume_skills),
                "status_reason": "No eligible resume evidence",
            }
            for index, requirement in enumerate(requirement_texts)
        ]
        return {
            "semantic_score": 0.0,
            "strong_matches": [],
            "partial_matches": [],
            "gaps": pairs,
            "ignored_lines": ignored_lines,
            "requirements": requirement_records,
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
    coverage_credits = []
    coverage_weights = []
    for index, requirement in enumerate(requirement_texts):
        items = embedded_evidence[index]
        requirement_skills = set(extract_technical_skills(requirement))
        missing_skills = sorted(requirement_skills - resume_skills)
        if not items:
            pair = {
                "requirement": requirement,
                "evidence": "",
                "section": "",
                "score": 0.0,
                "raw_score": 0.0,
                "status": "Gap",
                "optional": requirement_records[index].get("optional", False),
                "missing_skills": missing_skills,
                "status_reason": (
                    f"{', '.join(missing_skills)} not found in resume"
                    if missing_skills else "No eligible resume evidence"
                ),
            }
            gaps.append(pair)
            raw_match_scores.append(pair)
            coverage_credits.append(0.0)
            coverage_weights.append(0.5 if requirement_records[index].get("optional") else 1.0)
            continue
        item_vectors = passage_vectors[offset:offset + len(items)]
        offset += len(items)
        row = cosine_similarity(query_vectors[index:index + 1], item_vectors)[0]
        best_index = int(np.argmax(row))
        raw_score = float(np.clip(row[best_index], 0.0, 1.0))
        status = classify_similarity(raw_score)
        status_reason = ""
        if missing_skills:
            if not requirement_skills.intersection(resume_skills):
                status = "Gap"
                status_reason = f"{', '.join(missing_skills)} not found in resume"
            else:
                status = "Partial"
                status_reason = f"Capped to Partial: {', '.join(missing_skills)} not found in resume"
        elif status == "Gap":
            status_reason = "Cosine below partial threshold"
        scaled_score = coverage_credit(raw_score, status)
        best_evidence = items[best_index]
        pair = {
            "requirement": requirement,
            "evidence": best_evidence["text"],
            "section": best_evidence["section"],
            "score": round(scaled_score, 4),
            "raw_score": round(raw_score, 4),
            "status": status,
            "optional": requirement_records[index].get("optional", False),
            "missing_skills": missing_skills,
            "status_reason": status_reason,
        }
        raw_match_scores.append(pair)
        coverage_credits.append(scaled_score)
        coverage_weights.append(0.5 if requirement_records[index].get("optional") else 1.0)
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
        "semantic_score": float(np.average(coverage_credits, weights=coverage_weights)),
        "strong_matches": strong_matches,
        "partial_matches": partial_matches,
        "gaps": gaps,
        "ignored_lines": ignored_lines,
        "requirements": requirement_records,
        "raw_match_scores": raw_match_scores,
    }