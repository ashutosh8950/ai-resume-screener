import os
from pathlib import Path

import pytest

from resume_screener import analyzer as analyzer_module, embeddings
from resume_screener.analyzer import ResumeAnalyzer


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
        "django", "react", "linux", "postgresql", "mysql", "aws", "azure", "gcp"
    ))
    assert "git and github" in strong_text
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
    assert "compensation" in ignored_reasons


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
    assert ignored == [
        {"line": "Preferred", "reason": "section heading"},
        {"line": "Good to have", "reason": "section heading"},
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
        "PostgreSQL MySQL GridFS JavaScript TypeScript LinkedIn GitHub\n"
        "I am an A student \ue000\ufffd\u200b\x01\u25af"
    )
    normalized = embeddings.normalize_text(raw)

    assert "Developer Tools" in normalized
    assert "Python\nCoding Ninjas Link" in normalized
    assert "Languages: Python" in normalized
    assert "Built REST API" in normalized
    assert "Artificial Intelligence \u2014 Caf\u00e9." in normalized
    assert "PyTorch FastAPI OpenCV MongoDB PostgreSQL MySQL GridFS JavaScript TypeScript LinkedIn GitHub" in normalized
    assert "I am an A student" in normalized
    assert all(char not in normalized for char in ("\ue000", "\ufffd", "\u200b", "\x01", "\u25af"))


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
