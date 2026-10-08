from resume_screener import analyzer as analyzer_module, embeddings
from resume_screener.analyzer import ResumeAnalyzer
from resume_screener.config import (
    FALLBACK_KEYWORD_WEIGHT,
    FALLBACK_TFIDF_WEIGHT,
    KEYWORD_WEIGHT,
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


def test_requirement_matching_uses_best_resume_evidence(monkeypatch):
    class MockEmbeddingModel:
        def embed(self, texts):
            for text in texts:
                lowered = text.lower()
                if "cloud security" in lowered:
                    yield [0.0, 1.0, 0.0]
                elif "kubernetes" in lowered:
                    yield [-1.0, 0.0, 0.0]
                else:
                    yield [1.0, 0.0, 0.0]

    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: MockEmbeddingModel())
    score, matched, gaps = embeddings.match_requirements(
        "Built Python APIs. Designed cloud security controls.",
        "Python development required. Cloud security experience. Kubernetes deployment.",
    )

    assert abs(score - (2 / 3)) < 1e-12
    assert len(matched) == 2
    assert matched[0]["evidence"] == "Built Python APIs."
    assert matched[1]["evidence"] == "Designed cloud security controls."
    assert len(gaps) == 1
    assert gaps[0]["requirement"] == "Kubernetes deployment."


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
        results["tfidf_score"] * FALLBACK_TFIDF_WEIGHT
        + results["keyword_score"] * FALLBACK_KEYWORD_WEIGHT,
        4,
    )
    assert results["embeddings_used"] is False
    assert "disabled" in results["embedding_message"]
    assert results["scoring_mode"] == "tfidf_keyword_fallback"


def test_analyze_embedding_failure_falls_back_cleanly(monkeypatch):
    def fail_matching(_resume, _jd):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(analyzer_module, "EMBEDDINGS_ENABLED", True)
    monkeypatch.setattr(analyzer_module.embeddings, "match_requirements", fail_matching)
    results = ResumeAnalyzer().analyze("Python developer", "Python role")

    assert results["overall_score"] == round(
        results["tfidf_score"] * FALLBACK_TFIDF_WEIGHT
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
        lambda _resume, _jd: (0.6, [{"requirement": "Python", "evidence": "Python work", "score": 0.6}], []),
    )
    results = ResumeAnalyzer().analyze("Python developer", "Python role")

    assert results["overall_score"] == round(
        0.6 * SEMANTIC_WEIGHT
        + results["keyword_score"] * KEYWORD_WEIGHT
        + results["tfidf_score"] * TFIDF_WEIGHT,
        4,
    )
    assert results["semantic_score"] == 0.6
    assert results["scoring_mode"] == "semantic_embeddings"