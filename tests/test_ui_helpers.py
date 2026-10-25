from resume_screener.ui_helpers import build_evidence_table_html


def test_evidence_table_html_is_compact_safe_and_complete():
    rendered = build_evidence_table_html("Strong", [{
        "requirement": "<Python>",
        "status": "Strong",
        "score": 0.96,
        "section": "Experience",
        "evidence": "Built <APIs>",
        "status_reason": "Great",
        "raw_score": 0.96,
    }])

    assert "\n" not in rendered
    assert "  <" not in rendered
    assert rendered.startswith('<div class="evidence-table-wrap"><table class="evidence-table">')
    assert "<table" in rendered and "</table></div>" in rendered
    assert "&lt;Python&gt;" in rendered
    assert "&lt;APIs&gt;" in rendered
    assert "width:96.0%" in rendered
    assert "<th>Note</th>" in rendered
