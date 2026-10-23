"""Plotly charts for single and batch comparisons."""

import plotly.graph_objects as go


SURFACE = "#100b1b"
ACCENT = "#a855f7"
GREEN = "#6ee7b7"
AMBER = "#fbbf24"
RED = "#f472b6"
PURPLE = "#e879f9"
SEMANTIC = "#c084fc"
TFIDF = "#7c8db5"
KEYWORDS = "#f472b6"
TEXT = "#f7f3ff"
TEXT_MUTED = "#a89bb8"


def _base(title=""):
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=13), x=0.02),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_MUTED, family="Segoe UI"),
        margin=dict(l=20, r=20, t=40, b=20),
    )


def _color(value):
    if value >= 65:
        return GREEN
    if value >= 40:
        return AMBER
    return RED


def plot_score_breakdown(tfidf: int, keyword: int, semantic: int, overall: int) -> go.Figure:
    labels = ["TF-IDF Similarity", "Keyword Match", "Semantic Match", "Overall Score"]
    values = [tfidf, keyword, semantic, overall]
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=[TFIDF, KEYWORDS, ACCENT, GREEN], line=dict(width=0)),
        text=[f"{value}%" for value in values],
        textposition="outside",
        textfont=dict(color=TEXT, size=12),
        cliponaxis=False,
    ))
    fig.update_layout(
        **_base("Score Breakdown"),
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, color=TEXT),
        bargap=0.35,
        height=240,
    )
    return fig


def plot_skill_chart(matched: list, missing: list) -> go.Figure:
    matched_count = len(matched)
    missing_count = len(missing)
    if matched_count == 0 and missing_count == 0:
        matched_count = missing_count = 1
    fig = go.Figure(go.Pie(
        labels=["Matched", "Missing"],
        values=[matched_count, missing_count],
        hole=0.6,
        marker=dict(colors=[GREEN, RED], line=dict(color=SURFACE, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT),
        hovertemplate="%{label}: %{value} skills<extra></extra>",
    ))
    fig.update_layout(
        **_base("Skill Coverage"),
        showlegend=False,
        height=240,
        annotations=[dict(
            text=f"{matched_count}/{matched_count + missing_count}",
            x=0.5,
            y=0.5,
            font=dict(size=18, color=TEXT),
            showarrow=False,
        )],
    )
    return fig


def plot_batch_chart(batch_results: list) -> go.Figure:
    full_titles = [result["title"] for result in batch_results]
    titles = [title.split("@", 1)[0].strip() for title in full_titles]
    series = [
        ("TF-IDF", [result["tfidf"] for result in batch_results], TFIDF),
        ("Keywords", [result["kw"] for result in batch_results], KEYWORDS),
        ("Semantic", [result["semantic"] for result in batch_results], ACCENT),
        ("Skill coverage", [result["skill_coverage"] for result in batch_results], AMBER),
        ("Overall", [result["overall"] for result in batch_results], GREEN),
    ]
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(go.Bar(
            name=name,
            x=titles,
            y=values,
            customdata=full_titles,
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{value}%" for value in values],
            textposition="outside",
            textfont=dict(size=10, color=TEXT),
            cliponaxis=False,
            hovertemplate="%{customdata}<br>%{y}%<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        **_base("Job Comparison"),
        barmode="group",
        xaxis=dict(showgrid=False, color=TEXT, tickfont=dict(size=11)),
        yaxis=dict(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            x=0,
            font=dict(color=TEXT_MUTED, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=300,
        bargap=0.2,
        bargroupgap=0.05,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=70))
    return fig


def plot_score_gauge(score: int, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="%", font=dict(color=color, size=32)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_MUTED, tickfont=dict(color=TEXT_MUTED)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=SURFACE,
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="#1a1a28"),
                dict(range=[40, 65], color="#1e1e2e"),
                dict(range=[65, 100], color="#1a2030"),
            ],
        ),
        title=dict(text=label, font=dict(color=TEXT_MUTED, size=11)),
    ))
    fig.update_layout(**_base(), height=180)
    return fig
