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
_EXPLICIT_EXCLUDED_LINE = re.compile(
    r"\b(?:benefits?|salary|compensation|pay range|location|relocation|"
    r"working days?|working hours?|work schedule|holidays?|paid time off|vacation|"
    r"health insurance|retirement contributions|equal opportunity|remote within)\b",
    re.IGNORECASE,
)
_SENIORITY_LINE = re.compile(r"^\s*\d+\s*(?:to|[-–])\s*\d+\s+years?\b", re.IGNORECASE)
_INLINE_HEADING = re.compile(r"^\s*([^:]{1,48}):\s*(.*)$")
_EXAMPLE_LIST = re.compile(r"\b(?:such as|e\.g\.?|for example)\s+(.+)$", re.IGNORECASE)
_REQUIREMENT_CUE = re.compile(
    r"\b(?:responsibilit(?:y|ies)|require(?:ment|ments|d|s)|qualif(?:ication|ications|ied)|"
    r"experience|proficien(?:t|cy)|knowledge|ability|skills?|must have|should have|"
    r"develop|build|design|implement|maintain|deploy|manage|support|lead|create|"
    r"bachelor|degree|years?|interested|use|familiarity|exposure|understanding)\b",
    re.IGNORECASE,
)
_EDUCATION_CUE = re.compile(r"\b(?:bachelor|master|phd|degree|diploma|coursework|education)\b", re.IGNORECASE)
_POSITIVE_HEADINGS = {
    "requirements", "key responsibilities", "responsibilities", "skills",
    "other important skills", "technical skills", "qualifications", "what you will do",
    "what we expect", "the candidate should be interested in", "required qualifications",
    "minimum qualifications", "preferred qualifications", "required skills", "required experience",
    "what you'll bring", "what you will bring", "what we're looking for", "what we are looking for",
    "tools", "tools and technologies", "technologies", "programming languages",
    "preferred skills", "desired skills", "must have", "nice to have", "good to have",
    "your qualifications", "what we need",
}
_OPTIONAL_HEADINGS = {
    "preferred", "preferred qualifications", "preferred skills", "preferred experience",
    "nice to have", "good to have", "optional", "bonus skills", "desired qualifications",
}
_NEGATIVE_HEADINGS = {
    "about us", "about the company", "about our company", "about the role", "job description",
    "additional information", "benefits", "perks", "compensation", "salary", "location",
    "work location", "working hours", "work schedule", "employment type", "equal opportunity",
}
_RESUME_HEADINGS = {
    "summary", "professional summary", "profile", "objective", "experience",
    "professional experience", "work experience", "employment history", "work history",
    "skills", "technical skills", "core skills", "core competencies", "developer tools",
    "frameworks", "databases", "languages", "technologies", "education", "projects", "project experience",
    "personal projects", "certifications", "certification", "licenses", "awards", "publications",
    "volunteer experience", "references", "achievements", "career highlights", "key skills",
    "technical competencies", "selected projects", "relevant experience", "career objective",
    "professional development", "coursework", "training",
}
_SKILL_HEADINGS = {
    "skills", "technical skills", "core skills", "core competencies", "developer tools",
    "frameworks", "databases", "languages", "technologies", "tools", "coursework",
}
_CERTIFICATION_HEADINGS = {"certifications", "certification", "licenses", "training"}
_SKILL_TERMS = sorted({
    entry["name"] for category in SKILL_CATEGORIES.values() for entry in category
} | {
    alias for category in SKILL_CATEGORIES.values() for entry in category for alias in entry["aliases"]
}, key=len, reverse=True)
_SKILL_TERM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(term) for term in _SKILL_TERMS) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_ALTERNATIVE_SEPARATOR = re.compile(r"\s*(?:,\s*(?:or\s*)?|/|\bor\b)\s*", re.IGNORECASE)
_RESUME_LIST_PREFIX = re.compile(
    r"^\s*(languages|developer tools|core skills|skills|technical skills|core competencies|"
    r"frameworks|databases|coursework|tools|technologies|certifications|certification)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_METRIC_FRAGMENT = re.compile(r"^\d+(?:\.\d+)?\s+[A-Z][\w-]*(?:\s+[-\w-]+)*\.?$")
_CERTIFICATION_TOPIC = re.compile(
    r"\b(?:certification|certified|certificate|credential|coursework|training|learned|studied|course)\b",
    re.IGNORECASE,
)
_STRAY_INITIAL = re.compile(r"(?<!\S)([B-HJ-Z])\s+([a-z])")
_CAMEL_CASE_EXCEPTIONS = {
    "linkedin", "github", "javascript", "typescript", "pytorch", "fastapi",
    "opencv", "mongodb", "postgresql", "mysql", "gridfs", "pypdf2", "pypdf",
    "chatgpt", "githubcopilot", "gpt35", "scikitlearn", "easyocr", "pymupdf",
    "pymongo", "numpy", "spacy", "shieldai", "standardscaler", "openai",
    "tensorflow", "sqlite",
}
_LINK_LABEL_ARTIFACTS = {"link", "linkedin", "github", "portfolio", "coursera", "certificate", "remote"}
_CAMEL_CASE_PRODUCT_NAME = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*[A-Z]{2,}$")
_PLAIN_WORD_HEAD = re.compile(r"^[A-Z][a-z]+$")
_NUMERIC_PIPE_SIDE = re.compile(
    r"^(?:\d+(?:\.\d+)?%?|(?:19|20)\d{2}(?:\s*[-–]\s*(?:(?:19|20)?\d{2}))?)$"
)
for _skill_category in SKILL_CATEGORIES.values():
    for _skill_entry in _skill_category:
        for _skill_name in [_skill_entry["name"], *_skill_entry["aliases"]]:
            _compact_skill_name = re.sub(r"[^A-Za-z0-9+#]", "", _skill_name).lower()
            if _compact_skill_name:
                _CAMEL_CASE_EXCEPTIONS.add(_compact_skill_name)


def _split_camel_case(word: str) -> str:
    trailing_match = re.search(r"[.,;:!?)]*$", word)
    trailing = trailing_match.group(0) if trailing_match else ""
    core = word[:-len(trailing)] if trailing else word
    compact = re.sub(r"[^A-Za-z0-9+#]", "", core).lower()
    if compact in _CAMEL_CASE_EXCEPTIONS or _CAMEL_CASE_PRODUCT_NAME.fullmatch(core):
        return word

    for index, char in enumerate(core):
        if (
            index
            and core[index - 1].islower()
            and char.isupper()
            and _PLAIN_WORD_HEAD.fullmatch(core[:index])
            and core[index:].lower() in _LINK_LABEL_ARTIFACTS
        ):
            return f"{core[:index]} {core[index:]}{trailing}"
    return word


def _split_pipe_line(line: str) -> list[str]:
    segments = []
    start = 0
    for match in re.finditer(r"\s*\|\s*", line):
        left = line[start:match.start()].strip()
        right = line[match.end():].strip()
        if _NUMERIC_PIPE_SIDE.fullmatch(left) or _NUMERIC_PIPE_SIDE.fullmatch(right):
            continue
        segments.append(line[start:match.start()].strip())
        start = match.end()
    segments.append(line[start:].strip())
    return [segment for segment in segments if segment]


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
        line = re.sub(r"(?<=\d)\s*\|\s*(?=\d)", " | ", line)
        for pipe_segment in _split_pipe_line(line):
            for segment in pipe_segment.splitlines():
                segment = re.sub(r"\bTechnologyPatiala\b", "Technology Patiala", segment, flags=re.IGNORECASE)
                segment = re.sub(r"\bRewariRewari\b", "Rewari Rewari", segment, flags=re.IGNORECASE)
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
    has_sentence_punctuation = bool(re.search(r"[.!?]$", line.strip()))
    has_digits = any(char.isdigit() for char in stripped)
    short_heading = (
        _word_count(stripped) <= 8
        and not has_sentence_punctuation
        and not has_digits
    )
    if short_heading:
        if key == "education":
            return True, "education", True
        if key in _OPTIONAL_HEADINGS:
            return True, "requirements", True
        if key in _POSITIVE_HEADINGS:
            return True, "requirements", False
        if key in _NEGATIVE_HEADINGS:
            return True, "about_us" if key.startswith("about ") else "excluded", False
    heading_shape = line.strip().endswith(":") or stripped.isupper() or stripped.istitle()
    if (
        short_heading
        and heading_shape
        and not extract_skills(stripped)
    ):
        return True, "unknown", False
    return False, "", False


def technical_skill_groups(text: str) -> list[list[str]]:
    example_match = _EXAMPLE_LIST.search(text)
    searchable_text = example_match.group(1) if example_match else text
    matches = []
    for match in _SKILL_TERM_PATTERN.finditer(searchable_text):
        skills = extract_technical_skills(match.group(0))
        if skills:
            matches.append((match, skills[0]))
    if not matches:
        return []
    if example_match:
        return [list(dict.fromkeys(skill for _, skill in matches))]

    groups = []
    current = [matches[0][1]]
    previous_match = matches[0][0]
    for match, skill in matches[1:]:
        separator = searchable_text[previous_match.end():match.start()]
        if _ALTERNATIVE_SEPARATOR.fullmatch(separator):
            current.append(skill)
        else:
            groups.append(list(dict.fromkeys(current)))
            current = [skill]
        previous_match = match
    groups.append(list(dict.fromkeys(current)))
    return groups


def split_skill_alternatives(text: str) -> list[str]:
    """Compatibility helper; alternatives stay together as one requirement."""
    return [normalize_text(text)]


def filter_job_description(job_description: str) -> tuple[list[dict], list[dict]]:
    requirements = []
    ignored = []
    section = "unknown"
    optional_section = False
    for raw_line in job_description.splitlines():
        line = normalize_text(raw_line).strip()
        if not line:
            continue
        inline_heading = _INLINE_HEADING.match(line)
        if inline_heading:
            heading_label = inline_heading.group(1).strip()
            heading_text = inline_heading.group(2).strip()
            heading_key = _heading_key(heading_label)
            if heading_key in _NEGATIVE_HEADINGS or heading_key in {
                "experience", "job title", "about", "about us", "salary", "employment type"
            }:
                section = "about_us" if heading_key.startswith("about") else "excluded"
                optional_section = False
                reason = "about us section" if section == "about_us" else "excluded inline heading"
                if heading_key == "experience" and _SENIORITY_LINE.match(heading_text):
                    reason = "seniority information"
                elif _EXPLICIT_EXCLUDED_LINE.search(line):
                    reason = "explicit benefits, salary, location, or schedule line"
                ignored.append({"line": line, "reason": reason})
                continue
            if heading_key == "education":
                section = "education"
                optional_section = True
                if heading_text:
                    line = heading_text
                else:
                    ignored.append({"line": line, "reason": "section heading"})
                    continue
            elif heading_key in _POSITIVE_HEADINGS or heading_key in _OPTIONAL_HEADINGS:
                section = "requirements"
                optional_section = heading_key in _OPTIONAL_HEADINGS
                if heading_text:
                    line = heading_text
                else:
                    ignored.append({"line": line, "reason": "section heading"})
                    continue
        is_heading, heading_kind, heading_optional = _is_heading(line)
        if is_heading:
            section = heading_kind
            optional_section = heading_optional
            ignored.append({"line": line, "reason": "section heading"})
            continue
        if _SENIORITY_LINE.match(line):
            ignored.append({"line": line, "reason": "seniority information"})
            continue
        line_skills = extract_skills(line)
        if _EXPLICIT_EXCLUDED_LINE.search(line):
            ignored.append({"line": line, "reason": "explicit benefits, salary, location, or schedule line"})
            continue
        if section == "about_us":
            ignored.append({"line": line, "reason": "about us section"})
            continue
        if section == "excluded":
            ignored.append({"line": line, "reason": "non-requirement section"})
            continue
        if _word_count(line) < 4 and not line_skills and not optional_section:
            ignored.append({"line": line, "reason": "fewer than four words"})
            continue
        requirement_like = section in {"requirements", "education"} or bool(_REQUIREMENT_CUE.search(line))
        requirement_like = requirement_like or bool(line_skills) or optional_section
        if section == "education":
            requirement_like = bool(_EDUCATION_CUE.search(line)) or bool(line_skills)
        if not requirement_like:
            ignored.append({"line": line, "reason": "not requirement-focused"})
            continue
        for sentence in split_sentences(line):
            part = normalize_text(sentence)
            part_skills = extract_skills(part)
            example_match = _EXAMPLE_LIST.search(part)
            is_optional = optional_section or section == "education" or bool(example_match)
            if _word_count(part) >= 4 or part_skills or is_optional:
                examples = technical_skill_groups(part)[0] if example_match and technical_skill_groups(part) else []
                requirements.append({
                    "text": part,
                    "optional": is_optional,
                    "skill_groups": technical_skill_groups(part),
                    "examples": examples,
                })
            else:
                ignored.append({"line": part, "reason": "sentence has fewer than four words"})
    return requirements, ignored


def _is_clean_resume_heading(line: str) -> bool:
    return (
        _word_count(line) <= 6
        and not any(char.isdigit() for char in line)
        and not re.search(r"[.!?;,]", line)
    )


def _resume_heading(line: str) -> tuple[bool, str, bool, bool, str]:
    normalized = normalize_text(line).strip()
    inline = _RESUME_LIST_PREFIX.match(normalized)
    if inline:
        section = inline.group(1).strip()
        key = _heading_key(section)
        return (
            True,
            section,
            key in _SKILL_HEADINGS,
            key in _CERTIFICATION_HEADINGS,
            inline.group(2).strip(),
        )

    stripped = normalized.rstrip(":").strip()
    key = _heading_key(stripped)
    if key in _RESUME_HEADINGS and _is_clean_resume_heading(stripped):
        return (
            True,
            stripped,
            key in _SKILL_HEADINGS,
            key in _CERTIFICATION_HEADINGS,
            "",
        )
    return False, "", False, False, ""


def _bounded_evidence_chunks(text: str, max_chars: int = 300) -> list[str]:
    sentences = split_sentences(text) or [text.strip()]
    chunks = []
    current = ""
    for sentence in sentences:
        words = sentence.split()
        fragments = []
        fragment = ""
        for word in words:
            candidate = f"{fragment} {word}".strip()
            if len(candidate) > max_chars and fragment:
                fragments.append(fragment)
                fragment = word
            else:
                fragment = candidate
        if fragment:
            fragments.append(fragment)
        for fragment in fragments:
            candidate = f"{current} {fragment}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = fragment
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _append_segment(segments: list[dict], text: str, section: str, skills_only: bool, certifications_only: bool) -> None:
    for chunk in _bounded_evidence_chunks(text):
        if not chunk:
            continue
        segments.append({
            "text": chunk,
            "section": section,
            "skills_only": skills_only,
            "certifications_only": certifications_only,
        })


def segment_resume(resume: str) -> list[dict]:
    segments = []
    section = "Resume"
    skills_section = False
    certifications_section = False
    section_started = False
    for raw_line in resume.splitlines():
        is_bullet = bool(re.match(
            rf"^\s*(?:[{re.escape(_BULLET_CHARS)}]|[-*+]\s+|\d{{1,2}}[.)]\s+)",
            raw_line,
        ))
        normalized = normalize_text(raw_line).strip()
        if not normalized:
            continue
        for line_index, normalized_line in enumerate(normalized.splitlines()):
            is_heading, heading, inline_skills, inline_certifications, inline_content = _resume_heading(normalized_line)
            if is_heading:
                section = heading
                heading_key = _heading_key(heading)
                skills_section = inline_skills or heading_key in _SKILL_HEADINGS
                certifications_section = inline_certifications or heading_key in _CERTIFICATION_HEADINGS
                section_started = True
                normalized_line = inline_content
                if not normalized_line:
                    continue

            if not section_started:
                continue

            text = normalized_line.strip()
            if not text:
                continue
            metric_fragment = bool(_METRIC_FRAGMENT.fullmatch(text))
            can_merge = (
                bool(segments)
                and segments[-1]["section"] == section
                and not is_bullet
                and not skills_section
                and not certifications_section
                and (
                    metric_fragment
                    or not re.search(r"[.!?;:]$", segments[-1]["text"])
                )
            )
            if can_merge and len(segments[-1]["text"]) + len(text) + 1 <= 300:
                segments[-1]["text"] += " " + text
            else:
                _append_segment(segments, text, section, skills_section, certifications_section)
            if line_index > 0:
                is_bullet = False
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


def _record_skill_groups(record: dict) -> list[list[str]]:
    return record.get("skill_groups") or technical_skill_groups(record["text"])


def _missing_group_skills(groups: list[list[str]], resume_skills: set[str]) -> list[str]:
    return list(dict.fromkeys(
        skill
        for group in groups
        if not set(group).intersection(resume_skills)
        for skill in group
    ))


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
    requirement_skill_groups = [_record_skill_groups(item) for item in requirement_records]
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
    for requirement, skill_groups in zip(requirement_texts, requirement_skill_groups):
        certification_topic = bool(_CERTIFICATION_TOPIC.search(requirement))
        embedded_evidence.append([
            item for item in evidence
            if (not item["skills_only"] or bool(skill_groups))
            and (
                not item.get("certifications_only", False)
                or bool(skill_groups)
                or certification_topic
            )
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
                "missing_skills": _missing_group_skills(requirement_skill_groups[index], resume_skills),
                "status_reason": (
                    f"examples: {', '.join(requirement_records[index].get('examples', []))}"
                    if requirement_records[index].get("examples")
                    else "No eligible resume evidence"
                ),
                "examples": requirement_records[index].get("examples", []),
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
    coverage_credits = []
    coverage_weights = []
    for index, requirement in enumerate(requirement_texts):
        items = embedded_evidence[index]
        skill_groups = requirement_skill_groups[index]
        missing_skills = _missing_group_skills(skill_groups, resume_skills)
        matched_skill_groups = [group for group in skill_groups if set(group).intersection(resume_skills)]
        examples = requirement_records[index].get("examples", [])
        if not items:
            pair = {
                "requirement": requirement,
                "evidence": "",
                "section": "",
                "score": 0.0,
                "raw_score": 0.0,
                "status": "Gap",
                "optional": requirement_records[index].get("optional", False),
                "missing_skills": [] if requirement_records[index].get("optional") else missing_skills,
                "status_reason": (
                    f"examples: {', '.join(examples)}"
                    if examples
                    else f"{', '.join(missing_skills)} not found in resume"
                    if missing_skills else "No eligible resume evidence"
                ),
                "examples": examples,
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
        generic_sql_mismatch = (
            any(set(group).intersection({"postgresql", "mysql"}) for group in skill_groups)
            and "sql" in resume_skills
            and not resume_skills.intersection({"postgresql", "mysql"})
        )
        if missing_skills:
            if generic_sql_mismatch:
                if status == "Strong":
                    status = "Partial"
                status_reason = "SQL present, PostgreSQL/MySQL not found"
            elif not matched_skill_groups:
                status = "Gap"
                status_reason = f"{', '.join(missing_skills)} not found in resume"
            else:
                status = "Partial"
                status_reason = f"Capped to Partial: {', '.join(missing_skills)} not found in resume"
        elif status == "Gap":
            status_reason = "Cosine below partial threshold"
        if examples:
            status_reason = f"examples: {', '.join(examples)}"
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
            "missing_skills": [] if requirement_records[index].get("optional") else missing_skills,
            "status_reason": status_reason,
            "examples": examples,
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