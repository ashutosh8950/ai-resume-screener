"""
visualizer.py  (v2)
Plotly charts — updated for 4 scores + batch mode bar chart.
"""

import plotly.graph_objects as go

SURFACE    = "#13131a"
ACCENT     = "#7c6bff"
GREEN      = "#3ddfa0"
AMBER      = "#f5c842"
RED        = "#ff6b6b"
PURPLE     = "#b8a9ff"
TEXT       = "#f0eeff"
TEXT_MUTED = "#9b96c9"


def _base(title=""):
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=13), x=0.02),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_MUTED, family="Segoe UI"),
        margin=dict(l=20, r=20, t=40, b=20),
    )


def _color(v):
    if v >= 70: return GREEN
    if v >= 45: return AMBER
    return RED


def plot_score_breakdown(tfidf: int, kw: int, w2v: int, overall: int) -> go.Figure:
    """Horizontal bar chart — now includes Word2Vec score."""
    labels = ["TF-IDF Similarity", "Keyword Match", "Word2Vec Semantic", "Overall Score"]
    values = [tfidf, kw, w2v, overall]
    colors = [_color(v) for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT, size=12),
        cliponaxis=False,
    ))
    fig.update_layout(
        **_base("Score Breakdown"),
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, color=TEXT),
        bargap=0.35, height=240,
    )
    return fig


def plot_skill_chart(matched: list, missing: list) -> go.Figure:
    """Donut chart — matched vs missing skills."""
    m    = max(len(matched), 0)
    miss = max(len(missing), 0)
    if m == 0 and miss == 0:
        m = miss = 1

    fig = go.Figure(go.Pie(
        labels=["Matched", "Missing"],
        values=[m, miss],
        hole=0.6,
        marker=dict(colors=[GREEN, RED], line=dict(color=SURFACE, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT),
        hovertemplate="%{label}: %{value} skills<extra></extra>",
    ))
    fig.update_layout(
        **_base("Skill Coverage"),
        showlegend=False, height=240,
        annotations=[dict(
            text=f"{m}/{m+miss}",
            x=0.5, y=0.5,
            font=dict(size=18, color=TEXT),
            showarrow=False
        )]
    )
    return fig


def plot_batch_chart(batch_results: list) -> go.Figure:
    """
    Grouped bar chart for batch mode.
    Shows TF-IDF, Keyword, Word2Vec, and Overall per job.
    """
    titles  = [r["title"][:28] + ("…" if len(r["title"]) > 28 else "") for r in batch_results]
    overall = [r["overall"] for r in batch_results]
    tfidf   = [r["tfidf"]   for r in batch_results]
    kw      = [r["kw"]      for r in batch_results]
    w2v     = [r["w2v"]     for r in batch_results]

    fig = go.Figure()
    for name, vals, color in [
        ("TF-IDF",    tfidf,   ACCENT),
        ("Keywords",  kw,      PURPLE),
        ("Word2Vec",  w2v,     "#5de0d4"),
        ("Overall",   overall, GREEN),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=titles, y=vals,
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{v}%" for v in vals],
            textposition="outside",
            textfont=dict(size=10, color=TEXT),
            cliponaxis=False,
        ))

    fig.update_layout(
        **_base("Job Comparison"),
        barmode="group",
        xaxis=dict(showgrid=False, color=TEXT, tickfont=dict(size=11)),
        yaxis=dict(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(color=TEXT_MUTED, size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        height=300,
        bargap=0.2,
        bargroupgap=0.05,
    )
    return fig


def plot_score_gauge(score: int, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="%", font=dict(color=color, size=32)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_MUTED, tickfont=dict(color=TEXT_MUTED)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=SURFACE, borderwidth=0,
            steps=[
                dict(range=[0, 45],   color="#1a1a28"),
                dict(range=[45, 70],  color="#1e1e2e"),
                dict(range=[70, 100], color="#1a2030"),
            ],
        ),
        title=dict(text=label, font=dict(color=TEXT_MUTED, size=11))
    ))
    fig.update_layout(**_base(), height=180)
    return fig
