from resume_screener import analyzer as analyzer_module
from resume_screener.analyzer import ResumeAnalyzer
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


def test_analyze_empty_input_is_safe(monkeypatch):
    monkeypatch.setattr(analyzer_module, "GENSIM_AVAILABLE", False)
    results = ResumeAnalyzer().analyze("", "")

    assert results["overall_score"] == 0.0
    assert results["w2v_status"] == "unavailable"
    assert results["w2v_used"] is False
    assert results["scoring_method"] == "tfidf_keyword_fallback"


def test_analyze_normal_input_reports_fallback(monkeypatch):
    monkeypatch.setattr(analyzer_module, "GENSIM_AVAILABLE", False)
    results = ResumeAnalyzer().analyze("Python developer with SQL", "Python and SQL required")

    assert results["tfidf_score"] > 0
    assert results["keyword_score"] > 0
    assert results["overall_score"] == round(
        results["tfidf_score"] * 0.5 + results["keyword_score"] * 0.5, 4
    )
    assert results["w2v_available"] is False
    assert "unavailable" in results["w2v_message"]


def test_analyze_preserves_existing_weighted_formula(monkeypatch):
    def fake_word2vec(self, _resume, _jd):
        self._w2v_status = "ok"
        return 0.5

    monkeypatch.setattr(ResumeAnalyzer, "word2vec_similarity", fake_word2vec)
    results = ResumeAnalyzer().analyze("Python developer", "Python role")

    expected = round(
        results["tfidf_score"] * 0.35
        + results["keyword_score"] * 0.35
        + 0.5 * 0.30,
        4,
    )
    assert results["overall_score"] == expected
    assert results["scoring_method"] == "weighted_with_word2vec"