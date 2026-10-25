from resume_screener.ui_helpers import build_leaderboard_html


def test_leaderboard_html_is_compact_and_preserves_rank_order():
    rows = [
        {"rank": 1, "file_name": "one<resume>.pdf", "overall_pct": 80, "semantic_pct": 81, "skill_coverage_pct": 79, "strong_count": 3, "partial_count": 1, "gaps_count": 0, "top_missing_skills": "Python"},
        {"rank": 2, "file_name": "two.pdf", "overall_pct": 55, "semantic_pct": 56, "skill_coverage_pct": 54, "strong_count": 2, "partial_count": 1, "gaps_count": 1, "top_missing_skills": "SQL"},
        {"rank": 3, "file_name": "three.pdf", "overall_pct": 30, "semantic_pct": 31, "skill_coverage_pct": 29, "strong_count": 1, "partial_count": 1, "gaps_count": 2, "top_missing_skills": "Docker"},
    ]
    rendered = build_leaderboard_html(rows)

    assert "\n" not in rendered
    assert rendered.count("<tr>") == len(rows) + 1
    assert rendered.index("#1") < rendered.index("#2") < rendered.index("#3")
    assert "one&lt;resume&gt;.pdf" in rendered
