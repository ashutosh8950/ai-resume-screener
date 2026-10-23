"""Small HTML builders used by the Streamlit presentation layer."""

import html


def build_evidence_table_html(title: str, rows: list[dict]) -> str:
    """Build one compact, safe HTML evidence table for ``st.markdown``."""
    status_colors = {"Strong": "#6ee7b7", "Partial": "#fbbf24", "Gaps": "#f472b6"}
    color = status_colors.get(title, "#a855f7")
    rendered_rows = []
    for pair in rows:
        note_parts = [pair.get("status_reason", ""), f"cosine {pair['raw_score']:.2f}"]
        if pair.get("optional"):
            note_parts.insert(0, "Optional · weight 0.5")
        note = ". ".join(part.strip().rstrip(".") for part in note_parts if part.strip())
        score = max(0, min(100, pair["score"] * 100))
        rendered_rows.append(
            f'<tr><td>{html.escape(str(pair["requirement"]))}</td>'
            f'<td class="evidence-score-cell">'
            f'<div class="evidence-score-label" style="color:{color}">'
            f'{html.escape(str(pair["status"]))} · {pair["score"]:.0%}</div>'
            f'<div class="evidence-score-bar"><span style="width:{score:.1f}%;background:{color}"></span></div>'
            f'</td><td>{html.escape(str(pair["section"]))}</td>'
            f'<td>{html.escape(str(pair["evidence"] or "No eligible resume evidence"))}</td>'
            f'<td>{html.escape(note)}</td></tr>'
        )
    return (
        '<div class="evidence-table-wrap"><table class="evidence-table">'
        '<thead><tr><th>Requirement</th><th>Score</th><th>Section</th><th>Best evidence</th><th>Note</th></tr></thead>'
        '<tbody>' + ''.join(rendered_rows) + '</tbody></table></div>'
    )


def build_leaderboard_html(rows: list[dict]) -> str:
    """Build the complete bulk leaderboard as one compact HTML string."""
    rendered_rows = []
    for row in rows:
        overall = int(row["overall_pct"])
        color = "#6ee7b7" if overall >= 65 else "#fbbf24" if overall >= 40 else "#f472b6"
        tier = "Strong" if overall >= 65 else "Moderate" if overall >= 40 else "Weak"
        missing = [skill.strip() for skill in row.get("top_missing_skills", "").split(",") if skill.strip()]
        missing_chips = " ".join(
            f"<span class='leaderboard-chip missing'>{html.escape(skill)}</span>" for skill in missing
        ) or "<span class='leaderboard-chip'>None</span>"
        metrics = " ".join([
            f"<span class='leaderboard-chip'>Semantic {row['semantic_pct']}%</span>",
            f"<span class='leaderboard-chip'>Skills {row['skill_coverage_pct']}%</span>",
            f"<span class='leaderboard-chip'>Strong {row['strong_count']}</span>",
            f"<span class='leaderboard-chip'>Partial {row['partial_count']}</span>",
            f"<span class='leaderboard-chip'>Gaps {row['gaps_count']}</span>",
        ])
        rendered_rows.append(
            f'<tr><td class="leaderboard-rank">#{row["rank"]}</td>'
            f'<td class="leaderboard-name">{html.escape(str(row["file_name"]))}</td>'
            f'<td class="leaderboard-score"><div style="color:{color}">{overall}% '
            f'<span style="font-size:.68rem;font-weight:600">{tier}</span></div>'
            f'<div class="leaderboard-score-bar"><span style="width:{max(0, min(100, overall))}%;background:{color}"></span></div></td>'
            f'<td>{metrics}</td><td>{missing_chips}</td></tr>'
        )
    return (
        '<div class="leaderboard-wrap"><table class="leaderboard-table">'
        '<thead><tr><th>Rank</th><th>Resume</th><th>Overall</th><th>Coverage and evidence</th><th>Top missing skills</th></tr></thead>'
        '<tbody>' + ''.join(rendered_rows) + '</tbody></table></div>'
    )
