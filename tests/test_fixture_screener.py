import os
from pathlib import Path

import pytest

from resume_screener import analyzer as analyzer_module, embeddings
from resume_screener.analyzer import ResumeAnalyzer
from resume_screener.suggestions import build_rule_suggestions


FIXTURE_DIR = Path(__file__).with_name("fixtures")
SAMPLE_RESUME = (FIXTURE_DIR / "sample_resume.txt").read_text(encoding="utf-8")
SAMPLE_JD = (FIXTURE_DIR / "sample_jd.txt").read_text(encoding="utf-8")


class MockEmbeddingModel:
    def embed(self, texts):
        for _ in texts:
            yield [1.0, 0.0]


def _use_mock_embeddings(monkeypatch):
    monkeypatch.setattr(analyzer_module, "EMBEDDINGS_ENABLED", True)
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: MockEmbeddingModel())


def test_fixture_analysis_gaps_missing_stacks_and_preserves_git_github(monkeypatch):
    _use_mock_embeddings(monkeypatch)

    result = ResumeAnalyzer().analyze(SAMPLE_RESUME, SAMPLE_JD)
    gap_text = " ".join(pair["requirement"].lower() for pair in result["gaps"])
    strong_text = " ".join(pair["requirement"].lower() for pair in result["strong_matches"])

    assert all(skill in result["missing_skills"] for skill in (
        "django", "react", "linux", "postgresql", "mysql", "aws", "azure", "gcp"
    ))
    assert all(term in gap_text for term in (
        "django", "react", "linux", "aws", "azure", "google cloud"
    ))
    assert "git and github" in strong_text
    assert all(skill in result["matched_skills"] for skill in (
        "docker", "python", "git", "github", "sql"
    ))
    assert "collaboration" in result["soft_skills_detected"]
    assert not any("collaboration" in pair["status_reason"].lower() for pair in result["strong_matches"])

    optional = next(
        pair for pair in result["raw_match_scores"]
        if "kubernetes" in pair["requirement"].lower()
    )
    assert optional["optional"] is True
    assert optional in result["gaps"]

    ignored_text = " ".join(item["line"].lower() for item in result["ignored_lines"])
    ignored_reasons = " ".join(item["reason"].lower() for item in result["ignored_lines"])
    assert "health insurance" in ignored_text
    assert "remote within the united states" in ignored_text
    assert "competitive annual compensation" in ignored_text
    assert "benefits" in ignored_reasons
    assert "location" in ignored_reasons
    assert "salary" in ignored_reasons

    postgresql_pair = next(
        pair for pair in result["raw_match_scores"]
        if pair["requirement"].lower().startswith("postgresql")
    )
    assert postgresql_pair["status_reason"] == "SQL present, PostgreSQL/MySQL not found"


def test_fixture_general_expectations_are_not_gaps_and_scores_are_unchanged(monkeypatch):
    _use_mock_embeddings(monkeypatch)

    result = ResumeAnalyzer().analyze(SAMPLE_RESUME, SAMPLE_JD)
    expectation_text = " ".join(item["requirement"].lower() for item in result["general_expectations"])
    gap_text = " ".join(item["requirement"].lower() for item in result["gaps"])
    suggestions = build_rule_suggestions(result)
    general_requirements = {
        item["requirement"].lower() for item in result["general_expectations"]
    }
    gap_suggestion_requirements = {
        item["requirement"].lower() for item in suggestions["gaps"]
    }

    assert "learning new technologies" in expectation_text
    assert "learning new technologies" not in gap_text
    assert not general_requirements.intersection(gap_suggestion_requirements)
    assert result["semantic_score"] == pytest.approx(0.49523809523809526)
    assert result["skill_coverage"] == pytest.approx(0.5454545454545454)
    assert result["keyword_score"] == pytest.approx(0.21818181818181817)
    assert result["tfidf_score"] == pytest.approx(0.07431279147973441)
    assert result["overall_score"] == pytest.approx(0.458)


def test_rule_suggestions_use_fixture_missing_partial_gap_and_low_keyword_sections(monkeypatch):
    _use_mock_embeddings(monkeypatch)
    analysis = ResumeAnalyzer().analyze(SAMPLE_RESUME, SAMPLE_JD)

    suggestions = build_rule_suggestions(analysis)

    assert suggestions["missing_skills"]
    assert all(item["skill"] and item["tip"] for item in suggestions["missing_skills"])
    assert suggestions["partial_matches"]
    assert all(item["evidence"] and item["suggestion"] for item in suggestions["partial_matches"])
    assert suggestions["gaps"]
    assert suggestions["ats_tips"] == [
        "Use standard headings such as Experience, Projects, Education, and Skills.",
        "Include relevant job-description keywords in context within truthful evidence, not as an isolated list.",
        "Quantify results with scale, time, reliability, users, or other meaningful measures.",
    ]


def test_fixture_has_complete_skill_line_requirements_and_separate_resume_entries():
    requirements, _ = embeddings.filter_job_description(SAMPLE_JD)
    required_text = " ".join(item["text"].lower() for item in requirements)
    segments = embeddings.segment_resume(SAMPLE_RESUME)
    certification_units = [item for item in segments if item["section"] == "Certifications"]
    metric_units = [item for item in segments if "0.99 ROC-AUC" in item["text"]]

    assert all(skill in required_text for skill in (
        "django", "postgresql", "mysql", "linux", "ubuntu", "docker", "aws", "azure", "google cloud", "github copilot"
    ))
    assert len(certification_units) == 2
    assert len(metric_units) == 1
    assert metric_units[0]["section"] == "PROJECTS"
    assert "Built a REST API" in metric_units[0]["text"]
    assert "0.99 ROC-AUC." in metric_units[0]["text"]
    assert any("Implemented SQL queries" in item["text"] for item in segments)
    assert "0.99 ROC-AUC." not in {item["section"] for item in segments}
    assert all("9999999999" not in item["text"] for item in segments)
    assert all("gmail.com" not in item["text"].lower() for item in segments)


def test_inline_about_and_seniority_lines_are_ignored_with_reasons():
    requirements, ignored = embeddings.filter_job_description(SAMPLE_JD)
    ignored_by_line = {item["line"]: item["reason"] for item in ignored}

    assert ignored_by_line[
        "About Us: Fossaro is the dedicated back-office partner for growing education companies."
    ] == "about us section"
    assert ignored_by_line[
        "As the operational backbone of Bounce Teachers, we coordinate administration and support."
    ] == "about us section"
    assert ignored_by_line["Experience: 0 to 2 years"] == "seniority information"
    assert all("bounce teachers" not in item["text"].lower() for item in requirements)


def test_skill_alternatives_and_example_tools_remain_grouped(monkeypatch):
    _use_mock_embeddings(monkeypatch)
    requirements, _ = embeddings.filter_job_description(SAMPLE_JD)
    postgres_requirements = [
        item for item in requirements
        if "postgresql" in item["text"].lower() or "mysql" in item["text"].lower()
    ]
    cloud_requirements = [
        item for item in requirements
        if "exposure to aws" in item["text"].lower()
    ]
    ai_tool_requirements = [
        item for item in requirements
        if "ai coding tools" in item["text"].lower()
    ]

    assert len(postgres_requirements) == 1
    assert postgres_requirements[0]["skill_groups"] == [["postgresql", "mysql"]]
    assert len(cloud_requirements) == 1
    assert cloud_requirements[0]["skill_groups"] == [["aws", "azure", "gcp"]]
    assert len(ai_tool_requirements) == 1
    assert ai_tool_requirements[0]["optional"] is True
    assert ai_tool_requirements[0]["examples"] == ["github copilot", "cursor", "claude", "chatgpt"]

    analysis = ResumeAnalyzer().analyze(SAMPLE_RESUME, SAMPLE_JD)
    assert not any(
        skill in analysis["required_technical_skills"]
        for skill in ai_tool_requirements[0]["examples"]
    )
    example_pair = next(
        pair for pair in analysis["raw_match_scores"]
        if "ai coding tools" in pair["requirement"].lower()
    )
    assert example_pair["status_reason"] == "examples: github copilot, cursor, claude, chatgpt"


def test_any_cloud_skill_satisfies_one_cloud_requirement(monkeypatch):
    _use_mock_embeddings(monkeypatch)
    resume = "TECHNICAL SKILLS\nDocker, Python\nPROJECTS\nBuilt Docker images for deployments."
    jd = "REQUIREMENTS\nExposure to AWS, Azure, or Google Cloud."

    result = ResumeAnalyzer().analyze(resume, jd)

    assert result["requirement_count"] == 1
    assert result["required_technical_skill_groups"] == [["aws", "azure", "gcp"]]
    assert result["skill_coverage"] == 0.0
    assert len(result["gaps"]) == 1
    assert result["missing_skills"] == ["aws", "azure", "gcp"]


def test_optional_requirement_has_half_weight_in_semantic_coverage(monkeypatch):
    _use_mock_embeddings(monkeypatch)
    result = embeddings.match_requirements(
        "TECHNICAL SKILLS\nPython, Flask\nEXPERIENCE\nBuilt Python services for customer applications.",
        "REQUIREMENTS\nBuild Python backend services for production applications.\n"
        "PREFERRED\nDeploy workloads using Kubernetes orchestration tools.",
    )

    assert result["semantic_score"] == pytest.approx(1.0 / 1.5)


def test_short_skill_lines_are_kept_even_without_requirement_cues():
    requirements, ignored = embeddings.filter_job_description(
        "Linux\nPostgreSQL/MySQL\nAWS/Azure/GCP\nPreferred\nReact\nGood to have\nDocker"
    )

    assert [item["text"] for item in requirements] == [
        "Linux", "PostgreSQL/MySQL", "AWS/Azure/GCP", "React", "Docker"
    ]
    assert [item["optional"] for item in requirements] == [False, False, False, True, True]
    assert requirements[1]["skill_groups"] == [["postgresql", "mysql"]]
    assert requirements[2]["skill_groups"] == [["aws", "azure", "gcp"]]
    assert ignored == [
        {"line": "Preferred", "reason": "section heading"},
        {"line": "Good to have", "reason": "section heading"},
    ]


def test_jd_heading_with_sentence_period_does_not_change_section_state():
    requirements, _ = embeddings.filter_job_description(
        "REQUIREMENTS\nBuild and maintain Linux application services.\n"
        "Education.\nAdminister Linux systems for production workloads."
    )

    assert [item["text"] for item in requirements] == [
        "Build and maintain Linux application services.",
        "Administer Linux systems for production workloads.",
    ]


def test_partial_cap_explains_missing_technical_skill(monkeypatch):
    _use_mock_embeddings(monkeypatch)
    result = embeddings.match_requirements(
        "TECHNICAL SKILLS\nPython, Flask\nEXPERIENCE\nBuilt Python services for web applications.",
        "REQUIREMENTS\nUse Python and React to build modern web applications.",
    )

    assert len(result["partial_matches"]) == 1
    assert result["partial_matches"][0]["status_reason"] == "Capped to Partial: react not found in resume"


def test_normalize_repairs_all_reported_text_artifacts():
    raw = (
        "Developer T ools | Python|Coding NinjasLink\nLanguages:Python\n"
        "\u2013Built REST API with Django\n"
        "Artificial\u00a0Intelligence \u2014 Caf\u00e9. PyTorch FastAPI OpenCV MongoDB "
        "PostgreSQL MySQL GridFS JavaScript TypeScript LinkedIn GitHub PyPDF2 pypdf "
        "ChatGPT GitHub Copilot GPT-3.5 scikit-learn Scikit-learn\n"
        "I am an A student \ue000\ufffd\u200b\x01\u25af"
    )
    normalized = embeddings.normalize_text(raw)

    assert "Developer Tools" in normalized
    assert "Python\nCoding Ninjas Link" in normalized
    assert "Languages: Python" in normalized
    assert "Built REST API" in normalized
    assert "Artificial Intelligence \u2014 Caf\u00e9." in normalized
    assert "PyTorch FastAPI OpenCV MongoDB PostgreSQL MySQL GridFS JavaScript TypeScript LinkedIn GitHub" in normalized
    assert "PyPDF2 pypdf ChatGPT GitHub Copilot GPT-3.5 scikit-learn Scikit-learn" in normalized
    assert "I am an A student" in normalized
    assert all(char not in normalized for char in ("\ue000", "\ufffd", "\u200b", "\x01", "\u25af"))


def test_normalize_keeps_product_casing_and_only_splits_known_pdf_labels():
    products = (
        "EasyOCR PyMuPDF PyMongo NumPy spaCy GridFS ShieldAI StandardScaler OpenAI "
        "TensorFlow PyTorch scikit-learn FastAPI JavaScript TypeScript MySQL PostgreSQL "
        "MongoDB SQLite"
    )
    normalized_products = embeddings.normalize_text(products)
    assert normalized_products == products

    assert embeddings.normalize_text("NinjasLink EngineerLinkedIn") == "Ninjas Link Engineer LinkedIn"


def test_normalize_preserves_numeric_pipe_but_splits_text_pipe():
    numeric_line = "Class X — 82.2% 2021 | 2019"

    assert embeddings.normalize_text(numeric_line) == numeric_line
    assert embeddings.normalize_text("Python|Coding Ninjas") == "Python\nCoding Ninjas"


def test_normalize_repairs_named_spacing_artifacts_without_changing_products_or_skills():
    assert embeddings.normalize_text("TechnologyPatiala RewariRewari 2021|2019") == (
        "Technology Patiala Rewari Rewari 2021 | 2019"
    )
    assert embeddings.normalize_text("OpenAI GPT-3.5 ChatGPT") == "OpenAI GPT-3.5 ChatGPT"


def test_fixture_analyzer_score_uses_phase_3a_2_weights(monkeypatch):
    _use_mock_embeddings(monkeypatch)
    result = ResumeAnalyzer().analyze(SAMPLE_RESUME, SAMPLE_JD)
    expected = (
        0.55 * result["semantic_score"]
        + 0.30 * result["skill_coverage"]
        + 0.075 * result["keyword_score"]
        + 0.075 * result["tfidf_score"]
    )

    assert result["overall_score"] == pytest.approx(expected, abs=1e-4)


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_SLOW_TESTS") != "1", reason="set RUN_SLOW_TESTS=1 to use the real BGE model")
def test_real_model_fixture_analysis_reports_scores_and_groups():
    result = ResumeAnalyzer().analyze(SAMPLE_RESUME, SAMPLE_JD)
    print({
        "overall": result["overall_score"],
        "semantic": result["semantic_score"],
        "skill": result["skill_coverage"],
        "keyword": result["keyword_score"],
        "tfidf": result["tfidf_score"],
        "counts": {
            "strong": len(result["strong_matches"]),
            "partial": len(result["partial_matches"]),
            "gaps": len(result["gaps"]),
        },
    })
    assert result["embeddings_used"] is True
    assert result["gaps"]
    git_pair = next(
        pair
        for pair in result["raw_match_scores"]
        if "git and github" in pair["requirement"].lower()
    )
    assert git_pair["status"] == embeddings.classify_similarity(git_pair["raw_score"])
    assert "collaboration" not in git_pair["status_reason"].lower()
