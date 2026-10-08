from resume_screener import analyzer as analyzer_module, embeddings
from resume_screener.analyzer import ResumeAnalyzer
from resume_screener.config import (
    FALLBACK_KEYWORD_WEIGHT,
    FALLBACK_SKILL_WEIGHT,
    FALLBACK_TFIDF_WEIGHT,
    KEYWORD_WEIGHT,
    SKILL_WEIGHT,
    SEMANTIC_WEIGHT,
    TFIDF_WEIGHT,
)
from resume_screener.skills import extract_skills


def test_tokenize_keeps_complete_punctuation_tokens():
    tokens = ResumeAnalyzer().tokenize("JavaScript, Java, C++, C#, .NET, Node.js")

    assert tokens == ["javascript", "java", "c++", "c#", ".net", "node.js"]


def test_keyword_score_uses_whole_tokens():
    analyzer = ResumeAnalyzer()

    assert analyzer.keyword_score("javascript developer", "java") == 0.0
    assert analyzer.keyword_score("java developer", "java") == 1.0


def test_tfidf_similarity_returns_similarity_for_matching_text():
    score = ResumeAnalyzer().tfidf_similarity("Python data analysis", "Python data analysis")

    assert abs(score - 1.0) < 1e-12


def test_extract_skills_supports_punctuation_and_aliases():
    skills = extract_skills("C++, C#, .NET, Node.js, JS, ML and JavaScript")

    assert "c++" in skills
    assert "c#" in skills
    assert ".net" in skills
    assert "node.js" in skills
    assert "javascript" in skills
    assert "machine learning" in skills
    assert "java" not in skills


def test_extract_skills_loads_categories_and_does_not_confuse_cpp_with_c():
    skills = extract_skills("C++, C#, .NET, Python, teamwork")

    assert "c++" in skills
    assert "c#" in skills
    assert ".net" in skills
    assert "python" in skills
    assert "teamwork" in skills
    assert "c" not in skills


def test_split_sentences_cleans_bullets_and_odd_symbols():
    sentences = embeddings.split_sentences(
        "• Built APIs.\n▯ Improved uptime.\n- Led delivery teams"
    )

    assert sentences == ["Built APIs.", "Improved uptime.", "Led delivery teams"]


def test_text_preprocessing_preserves_accents_dashes_punctuation_and_repairs_linkedin_space():
    cleaned = ResumeAnalyzer().preprocess(
        "Artificial Intelligence – end-to-end systems; Café, EngineerLinkedIn"
    )

    assert "artificial" in cleaned
    assert "–" in cleaned
    assert "end-to-end" in cleaned
    assert "café," in cleaned
    assert "engineer linkedin" in cleaned


def test_shared_normalizer_repairs_reported_pdf_and_paste_artifacts():
    text = embeddings.normalize_text(
        "Developer T ools | Python|Coding NinjasLink\n"
        "Languages:Python\n–Built REST API with Django\n"
        "Artificial Intelligence — Café. PyTorch FastAPI OpenCV MongoDB PostgreSQL MySQL GridFS"
    )

    assert "Developer Tools" in text
    assert "Python\nCoding Ninjas Link" in text
    assert "Languages: Python" in text
    assert "Built REST API" in text
    assert "Artificial Intelligence — Café." in text
    assert "PyTorch FastAPI OpenCV MongoDB PostgreSQL MySQL GridFS" in text


def test_job_description_filter_keeps_requirement_sections_and_reports_ignored_lines():
    requirements, ignored = embeddings.filter_job_description(
        """About Our Company:
We build helpful software for teams around the world.
Responsibilities:
- Build and maintain Django services for production teams.
Requirements:
- Strong communication and problem solving skills.
Benefits:
- Health insurance and paid vacation days.
Location: New York City office
Python, Django, PostgreSQL
"""
    )

    assert requirements == [
        {"text": "Build and maintain Django services for production teams.", "optional": False},
        {"text": "Strong communication and problem solving skills.", "optional": False},
        {"text": "Python", "optional": False},
        {"text": "Django", "optional": False},
        {"text": "PostgreSQL", "optional": False},
    ]
    ignored_text = [item["line"] for item in ignored]
    assert "About Our Company:" in ignored_text
    assert "Health insurance and paid vacation days." in ignored_text
    assert "Python, Django, PostgreSQL" not in ignored_text


def test_resume_segmentation_labels_sections_and_excludes_headings():
    segments = embeddings.segment_resume(
        """SUMMARY
Software engineer with five years of experience.
TECHNICAL SKILLS
Python, Django, Flask, SQL
EXPERIENCE
• Built Django APIs for internal teams
with Python and PostgreSQL.
EDUCATION
Bachelor of Engineering, 2022-2026
"""
    )

    assert all(item["text"] not in {"SUMMARY", "TECHNICAL SKILLS", "EXPERIENCE", "EDUCATION"} for item in segments)
    skills_line = next(item for item in segments if item["skills_only"])
    assert skills_line["section"] == "TECHNICAL SKILLS"
    experience = next(item for item in segments if "Built Django APIs" in item["text"])
    assert experience["section"] == "EXPERIENCE"
    assert "with Python and PostgreSQL" in experience["text"]


def test_skills_list_evidence_is_only_eligible_for_skill_requirements(monkeypatch):
    class MockEmbeddingModel:
        def embed(self, texts):
            for _ in texts:
                yield [1.0, 0.0]

    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: MockEmbeddingModel())
    result = embeddings.match_requirements(
        "Skills:\nPython, Django, Flask, SQL\n"
        "Experience:\n• Coordinated project timelines across design and delivery teams.",
        "Requirements:\nCoordinate project timelines across delivery teams effectively.\n"
        "Demonstrate Python programming experience for backend development.",
    )

    evidence_by_requirement = {
        item["requirement"]: item
        for item in result["strong_matches"]
    }
    assert evidence_by_requirement[
        "Coordinate project timelines across delivery teams effectively."
    ]["section"] == "Experience"
    assert evidence_by_requirement[
        "Demonstrate Python programming experience for backend development."
    ]["section"] == "Skills"


def test_requirement_matching_uses_best_resume_evidence(monkeypatch):
    embedded_texts = []

    class MockEmbeddingModel:
        def embed(self, texts):
            for text in texts:
                embedded_texts.append(text)
                lowered = text.lower()
                if "cloud security" in lowered:
                    yield [0.0, 1.0, 0.0]
                elif "kubernetes" in lowered:
                    yield [-1.0, 0.0, 0.0]
                else:
                    yield [1.0, 0.0, 0.0]

    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: MockEmbeddingModel())
    matched = embeddings.match_requirements(
        "Experience:\n• Built Python APIs for production teams.\n"
        "• Designed cloud security controls for enterprise systems.",
        "Requirements:\n- Python development experience required.\n"
        "- Cloud security engineering experience.\n"
        "- Kubernetes deployment experience required.",
    )

    assert abs(matched["semantic_score"] - (2 / 3)) < 1e-12
    assert len(matched["strong_matches"]) == 2
    assert matched["strong_matches"][0]["evidence"].startswith("Built Python APIs")
    assert matched["strong_matches"][1]["evidence"].startswith("Designed cloud security")
    assert matched["strong_matches"][0]["section"] == "Experience"
    assert len(matched["gaps"]) == 1
    assert matched["gaps"][0]["requirement"] == "Kubernetes deployment experience required."
    assert embedded_texts[0].startswith(embeddings.BGE_QUERY_PREFIX)
    assert all(not text.startswith(embeddings.BGE_QUERY_PREFIX) for text in embedded_texts[3:])


def test_skill_gate_marks_absent_framework_as_gap_even_with_high_cosine(monkeypatch):
    class MockEmbeddingModel:
        def embed(self, texts):
            for _ in texts:
                yield [1.0, 0.0]

    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: MockEmbeddingModel())
    result = embeddings.match_requirements(
        "Experience:\n• Built Flask REST APIs for internal services.",
        "Requirements:\n- Django REST Framework development experience required.",
    )

    assert result["strong_matches"] == []
    assert result["partial_matches"] == []
    assert result["gaps"][0]["missing_skills"] == ["django"]
    assert result["gaps"][0]["raw_score"] == 1.0


def test_soft_skills_do_not_cap_present_git_github_requirement(monkeypatch):
    class MockEmbeddingModel:
        def embed(self, texts):
            for _ in texts:
                yield [1.0, 0.0]

    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: MockEmbeddingModel())
    result = embeddings.match_requirements(
        "Experience:\n• Used Git and GitHub to review and merge team code.",
        "Requirements:\nUse Git and GitHub for team collaboration and code review.",
    )

    assert len(result["strong_matches"]) == 1
    assert result["strong_matches"][0]["missing_skills"] == []


def test_similarity_thresholds_group_strong_partial_and_gap():
    from resume_screener.config import PARTIAL_MATCH_THRESHOLD, STRONG_MATCH_THRESHOLD

    assert embeddings.classify_similarity(STRONG_MATCH_THRESHOLD) == "Strong"
    assert embeddings.classify_similarity(PARTIAL_MATCH_THRESHOLD) == "Partial"
    assert embeddings.classify_similarity(PARTIAL_MATCH_THRESHOLD - 0.01) == "Gap"
    assert embeddings.coverage_credit(PARTIAL_MATCH_THRESHOLD - 0.01, "Gap") == 0.0
    assert embeddings.coverage_credit(PARTIAL_MATCH_THRESHOLD, "Partial") == 0.40
    assert embeddings.coverage_credit(STRONG_MATCH_THRESHOLD, "Strong") == 0.80
    assert embeddings.coverage_credit(STRONG_MATCH_THRESHOLD + 0.15, "Strong") == 1.0


def test_partial_cosine_is_shown_in_partial_group(monkeypatch):
    class MockEmbeddingModel:
        def embed(self, texts):
            for text in texts:
                if text.startswith(embeddings.BGE_QUERY_PREFIX):
                    yield [1.0, 0.0]
                else:
                    yield [0.62, (1 - 0.62**2) ** 0.5]

    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: MockEmbeddingModel())
    result = embeddings.match_requirements(
        "Experience:\n• Coordinate work across product groups and support organizations.",
        "Requirements:\n- Collaborate with product stakeholders and support teams.",
    )

    assert result["strong_matches"] == []
    assert len(result["partial_matches"]) == 1
    assert result["gaps"] == []


def test_analyze_empty_input_is_safe(monkeypatch):
    monkeypatch.setattr(analyzer_module, "EMBEDDINGS_ENABLED", False)
    results = ResumeAnalyzer().analyze("", "")

    assert results["overall_score"] == 0.0
    assert results["embeddings_used"] is False
    assert results["scoring_mode"] == "tfidf_keyword_fallback"


def test_analyze_disabled_embeddings_uses_fallback_weights(monkeypatch):
    monkeypatch.setattr(analyzer_module, "EMBEDDINGS_ENABLED", False)
    results = ResumeAnalyzer().analyze("Python developer with SQL", "Python and SQL required")

    assert results["tfidf_score"] > 0
    assert results["keyword_score"] > 0
    assert results["overall_score"] == round(
        results["skill_coverage"] * FALLBACK_SKILL_WEIGHT
        + results["tfidf_score"] * FALLBACK_TFIDF_WEIGHT
        + results["keyword_score"] * FALLBACK_KEYWORD_WEIGHT,
        4,
    )
    assert results["embeddings_used"] is False
    assert "disabled" in results["embedding_message"]
    assert results["scoring_mode"] == "tfidf_keyword_fallback"


def test_analyze_embedding_failure_falls_back_cleanly(monkeypatch):
    def fail_matching(_resume, _jd, **_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(analyzer_module, "EMBEDDINGS_ENABLED", True)
    monkeypatch.setattr(analyzer_module.embeddings, "match_requirements", fail_matching)
    results = ResumeAnalyzer().analyze(
        "Python developer", "Python role development experience required"
    )

    assert results["overall_score"] == round(
        results["skill_coverage"] * FALLBACK_SKILL_WEIGHT
        + results["tfidf_score"] * FALLBACK_TFIDF_WEIGHT
        + results["keyword_score"] * FALLBACK_KEYWORD_WEIGHT,
        4,
    )
    assert results["scoring_mode"] == "tfidf_keyword_fallback"
    assert "could not be loaded or run" in results["embedding_message"]


def test_analyze_uses_configured_embedding_weights(monkeypatch):
    monkeypatch.setattr(analyzer_module, "EMBEDDINGS_ENABLED", True)
    monkeypatch.setattr(
        analyzer_module.embeddings,
        "match_requirements",
        lambda _resume, _jd, requirements=None, ignored_lines=None: {
            "semantic_score": 0.6,
            "strong_matches": [{"requirement": "Python", "evidence": "Python work", "score": 0.6}],
            "partial_matches": [],
            "gaps": [],
            "ignored_lines": ignored_lines or [],
            "requirements": requirements or [],
            "raw_match_scores": [],
        },
    )
    results = ResumeAnalyzer().analyze(
        "Python developer", "Python role development experience required"
    )

    assert results["overall_score"] == round(
        0.6 * SEMANTIC_WEIGHT
        + results["skill_coverage"] * SKILL_WEIGHT
        + results["keyword_score"] * KEYWORD_WEIGHT
        + results["tfidf_score"] * TFIDF_WEIGHT,
        4,
    )
    assert results["semantic_score"] == 0.6
    assert results["scoring_mode"] == "semantic_embeddings"


def test_analyze_scores_only_filtered_requirement_text(monkeypatch):
    monkeypatch.setattr(analyzer_module, "EMBEDDINGS_ENABLED", False)
    analyzer = ResumeAnalyzer()
    scored_jd = {}

    def capture_tfidf(_resume, jd):
        scored_jd["tfidf"] = jd
        return 0.25

    def capture_keyword(_resume, jd):
        scored_jd["keyword"] = jd
        return 0.5

    monkeypatch.setattr(analyzer, "tfidf_similarity", capture_tfidf)
    monkeypatch.setattr(analyzer, "keyword_score", capture_keyword)
    results = analyzer.analyze(
        "Python developer with Django experience",
        "About Us:\nWe make great software for millions.\n"
        "Requirements:\nBuild reliable Python web applications for customers.\n"
        "Benefits:\nUnlimited vacation and health insurance provided.",
    )

    assert scored_jd["tfidf"] == scored_jd["keyword"]
    assert scored_jd["tfidf"] == "Build reliable Python web applications for customers."
    assert results["requirement_count"] == 1
    assert results["raw_jd_token_count"] > results["jd_token_count"]