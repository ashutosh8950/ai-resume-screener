import csv
from io import StringIO

import pytest

from resume_screener.bulk import (
    CSV_COLUMNS,
    build_ranked_csv,
    check_upload_limits,
    rank_results,
    rank_batch_results,
    sanitize_csv_cell,
)


def test_check_upload_limits_skips_oversized_files_and_caps_file_count():
    files = [(f"resume-{index}.pdf", 1024) for index in range(11)]
    files[1] = ("too-large.pdf", 5 * 1024 * 1024 + 1)

    eligible, messages = check_upload_limits(files)

    assert eligible == [0, 2, 3, 4, 5, 6, 7, 8, 9]
    assert any("too-large.pdf" in message for message in messages)
    assert any("1 additional file" in message for message in messages)


@pytest.mark.parametrize(
    "value",
    ["=1+1", "+SUM(A1)", "-1+2", "@SUM(A1)", "\t=1+1", "\r=1+1"],
)
def test_sanitize_csv_cell_prefixes_formula_markers(value):
    assert sanitize_csv_cell(value) == f"'{value}"


def test_sanitize_csv_cell_preserves_safe_values():
    assert sanitize_csv_cell("resume.pdf") == "resume.pdf"
    assert sanitize_csv_cell(None) == ""


def test_rank_results_sorts_overall_score_and_assigns_ranks():
    rows = [
        {"file_name": "low.pdf", "overall_pct": 24},
        {"file_name": "high.pdf", "overall_pct": 91},
    ]

    ranked = rank_results(rows)

    assert [(row["rank"], row["file_name"]) for row in ranked] == [
        (1, "high.pdf"),
        (2, "low.pdf"),
    ]
    assert list(ranked[0])[0] == "rank"
    assert "rank" not in rows[0]


def test_rank_batch_results_breaks_rounded_overall_ties_precisely():
    rows = [
        {
            "title": "Semantic tie",
            "overall": 48,
            "_overall_score": 0.481,
            "skill_coverage": 0.70,
            "semantic": 0.80,
        },
        {
            "title": "Skill tie",
            "overall": 48,
            "_overall_score": 0.489,
            "skill_coverage": 0.60,
            "semantic": 0.90,
        },
    ]

    assert [row["title"] for row in rank_batch_results(rows)] == ["Skill tie", "Semantic tie"]


def test_build_ranked_csv_uses_ascii_headers_and_sanitizes_text_cells():
    csv_text = build_ranked_csv([{
        "rank": 1,
        "file_name": "=HYPERLINK(\"https://invalid\")",
        "overall_pct": 80,
        "semantic_pct": 70,
        "skill_coverage_pct": 60,
        "strong_count": 2,
        "partial_count": 1,
        "gaps_count": 3,
        "top_missing_skills": "@SUM(A1)",
        "mode": "semantic",
    }])
    rows = list(csv.DictReader(StringIO(csv_text)))

    assert tuple(rows[0]) == CSV_COLUMNS
    assert rows[0]["file_name"].startswith("'=")
    assert rows[0]["top_missing_skills"] == "'@SUM(A1)"
