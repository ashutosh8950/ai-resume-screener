"""
app.py  (v2 — upgraded)
Streamlit UI with three new features:
    1. PDF resume upload (pypdf)
    2. Pretrained semantic similarity score (FastEmbed ONNX)
    3. Batch mode — 1 resume vs multiple JDs
"""

import gc
import html

import streamlit as st

from resume_screener.analyzer import ResumeAnalyzer
from resume_screener.bulk import build_ranked_csv, check_upload_limits, rank_batch_results, rank_results
from resume_screener.config import (
    FALLBACK_KEYWORD_WEIGHT,
    FALLBACK_SKILL_WEIGHT,
    FALLBACK_TFIDF_WEIGHT,
    GEMINI_API_KEY,
    KEYWORD_WEIGHT,
    MAX_BATCH_JDS,
    MAX_TEXT_CHARS,
    MAX_UPLOAD_MB,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
    TFIDF_WEIGHT,
)
from resume_screener.embeddings import get_available_memory_mb
from resume_screener.logger import get_logger
from resume_screener.pdf_parser import PDFParseError, PDFParser
from resume_screener.suggestions import suggestions_for_analysis
from resume_screener.text_utils import clean_jd_text
from resume_screener.ui_helpers import build_evidence_table_html, build_leaderboard_html
from resume_screener.visualizer import plot_batch_chart, plot_score_breakdown, plot_skill_chart


logger = get_logger("app")


def _limit_text(text: str, label: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        st.warning(f"{label} exceeded {MAX_TEXT_CHARS:,} characters and was truncated.")
        return text[:MAX_TEXT_CHARS]
    return text


def _clean_jd_widget_state(key: str) -> None:
    """Normalize pasted or extracted JD text before Streamlit redraws the widget."""
    current = st.session_state.get(key, "")
    cleaned = clean_jd_text(current)
    if cleaned != current:
        st.session_state[key] = cleaned


def _normalize_jd_widget_state(key: str) -> None:
    """Clean an already-populated JD widget state before it is rendered."""
    if key in st.session_state:
        st.session_state[key] = clean_jd_text(st.session_state[key])


def _read_pdf(uploaded_file, label: str) -> tuple[str, int]:
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"{label} must be no larger than {MAX_UPLOAD_MB} MB.")
        return "", 0
    try:
        text, page_count = PDFParser().parse(uploaded_file.getvalue())
    except PDFParseError as error:
        st.error(str(error))
        return "", 0
    return _limit_text(text, label), page_count


def _render_match_group(title: str, pairs: list[dict], empty_message: str) -> None:
    status_class = title.lower().replace(" ", "-")
    st.markdown(f'<div class="evidence-heading result-reveal reveal-delay-3 {status_class}">{html.escape(title)} <span style="color:#8f849e;font-size:.7rem;font-weight:600;">{len(pairs)} found</span></div>', unsafe_allow_html=True)
    if not pairs:
        st.caption(empty_message)
        return

    def render_table(rows: list[dict]) -> None:
        st.markdown(build_evidence_table_html(title, rows), unsafe_allow_html=True)

    render_table(pairs[:8])

    if len(pairs) > 8:
        with st.expander(f"Show {len(pairs) - 8} more {title.lower()} matches"):
            render_table(pairs[8:])


def _render_bulk_leaderboard(rows: list[dict]) -> None:
    st.markdown(build_leaderboard_html(rows), unsafe_allow_html=True)

st.set_page_config(page_title="AI Resume Screener v2", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(124,58,237,.18), transparent 28rem),
        radial-gradient(circle at 92% 18%, rgba(232,121,249,.09), transparent 24rem),
        #08070d;
    color: #f7f3ff;
}
.stApp::before {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.22;
    background-image: linear-gradient(rgba(168,85,247,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(168,85,247,.045) 1px, transparent 1px);
    background-size:42px 42px;
    mask-image: linear-gradient(to bottom, black, transparent 86%);
    animation: gridDrift 24s linear infinite;
}
.stApp::after {
    content:""; position:fixed; width:28rem; height:28rem; top:9rem; right:-12rem; pointer-events:none; z-index:0;
    border-radius:50%; background:radial-gradient(circle, rgba(168,85,247,.16), transparent 68%);
    filter:blur(18px); animation: ambientFloat 10s ease-in-out infinite alternate;
}
.block-container { position:relative; z-index:1; max-width:1380px; padding-top:2.15rem; padding-bottom:3.4rem; }
h1,h2,h3,h4 { color:#f7f3ff; letter-spacing:-.025em; }
h2 { font-size:clamp(1.45rem, 2.5vw, 2.15rem)!important; }
hr { border-color:rgba(168,85,247,.18)!important; }
@keyframes gridDrift { from { background-position:0 0, 0 0; } to { background-position:0 42px, 42px 0; } }
@keyframes ambientFloat { from { transform:translate3d(0,0,0) scale(1); } to { transform:translate3d(-2rem,1.5rem,0) scale(1.08); } }
@property --scanner-angle { syntax:"<angle>"; inherits:false; initial-value:0deg; }
@keyframes scannerSpin { to { --scanner-angle:360deg; } }
@keyframes scannerDot { 0%,100% { transform:scale(.8); opacity:.7; } 50% { transform:scale(1.25); opacity:1; } }
@keyframes resultRingReveal { from { opacity:0; transform:scale(.82); } to { opacity:1; transform:scale(1); } }
@keyframes spinnerSweep { from { background-position:200% 0; } to { background-position:-200% 0; } }
@keyframes resultReveal { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.app-hero { position:relative; overflow:hidden; margin:0 0 1.25rem; padding:1.18rem 1.55rem 1.05rem; border:1px solid rgba(168,85,247,.28); border-radius:18px; background:linear-gradient(135deg, rgba(22,12,38,.92), rgba(12,9,20,.78)); box-shadow:0 0 46px rgba(124,58,237,.10), inset 0 1px rgba(255,255,255,.05); animation:heroRise .7s ease both; }
.app-hero::after { content:""; position:absolute; width:8rem; height:8rem; right:7%; top:1.15rem; border-radius:50%; background:conic-gradient(from var(--scanner-angle), #7c3aed, #a855f7, #e879f9, rgba(168,85,247,.2), #7c3aed); -webkit-mask:radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 2px)); mask:radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 2px)); box-shadow:0 0 0 1rem rgba(168,85,247,.035), 0 0 45px rgba(168,85,247,.2); animation:scannerSpin 8s linear infinite; }
.app-kicker { color:#e879f9; font:700 .68rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.65rem; }
.app-title { margin:0!important; font-size:clamp(1.8rem, 4vw, 3rem)!important; font-weight:800!important; }
.app-title span { color:#e879f9; text-shadow:0 0 22px rgba(232,121,249,.32); }
.app-subtitle { color:#aaa2ba; font-size:.9rem; line-height:1.7; margin:.5rem 0 0; max-width:880px; }
@keyframes heroRise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.section-kicker { color:#c084fc; font:700 .64rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.16em; text-transform:uppercase; margin:1.4rem 0 .35rem; }
.dashboard-heading { color:#f7f3ff; font-size:1.25rem; font-weight:750; margin:.05rem 0 .35rem; }
.dashboard-copy { color:#8f849e; font-size:.82rem; line-height:1.6; margin:0 0 1rem; }
.input-label { color:#e879f9; font:700 .68rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.14em; margin:.25rem 0 .55rem; }
.input-hint { color:#756b86; font-size:.72rem; line-height:1.5; margin:-.3rem 0 .7rem; }
.result-hero { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.4rem 0 1.1rem; padding:1rem 1.2rem; border:1px solid rgba(168,85,247,.3); border-radius:15px; background:linear-gradient(110deg, rgba(39,18,63,.78), rgba(17,11,28,.84)); box-shadow:0 14px 34px rgba(73,32,125,.14); }
.result-hero.tier-strong { --tier-color:#6ee7b7; }
.result-hero.tier-moderate { --tier-color:#fbbf24; }
.result-hero.tier-weak { --tier-color:#f472b6; }
.result-hero-label { color:#c084fc; font:700 .64rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.15em; text-transform:uppercase; }
.result-hero-title { color:#f7f3ff; font-size:1.08rem; font-weight:750; margin:.28rem 0 0; }
.result-hero-score { color:#e879f9; font-size:2rem; font-weight:850; line-height:1; text-shadow:0 0 24px rgba(232,121,249,.3); }
.result-hero-score small { display:block; color:#8f849e; font-size:.62rem; font-weight:600; letter-spacing:.08em; text-align:right; text-transform:uppercase; margin-top:.3rem; }
.result-gauge { display:grid; place-items:center; width:84px; height:84px; flex:0 0 84px; border-radius:50%; box-shadow:0 0 28px rgba(168,85,247,.2); }
.result-gauge-reveal { animation:resultRingReveal 1.2s ease-out both; animation-fill-mode:both; }
.result-gauge::after { content:""; grid-area:1/1; width:68px; height:68px; border-radius:50%; background:#130c20; box-shadow:inset 0 0 14px rgba(0,0,0,.45); }
.result-gauge-value { grid-area:1/1; z-index:1; color:#f7f3ff; font-size:1.05rem; font-weight:850; }
.sidebar-brand { padding:.2rem 0 1rem; border-bottom:1px solid rgba(168,85,247,.18); margin-bottom:1rem; }
.sidebar-brand-kicker { color:#c084fc; font:700 .6rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.14em; text-transform:uppercase; }
.sidebar-brand-title { color:#f7f3ff; font-size:1.12rem; font-weight:800; margin-top:.35rem; }
.sidebar-brand-title span { color:#e879f9; }
.sidebar-section-label { color:#8f849e; font:700 .62rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.13em; text-transform:uppercase; margin:.85rem 0 .45rem; }
.stRadio > div { gap:.35rem!important; }
.stRadio > label { border:0!important; border-radius:0!important; padding:0 0 .35rem!important; background:transparent!important; box-shadow:none!important; color:#b8acc8!important; font-size:.72rem!important; }
.stRadio label { border:1px solid rgba(168,85,247,.18); border-radius:9px; padding:.35rem .55rem!important; background:rgba(16,11,27,.5); color:#b8acc8!important; transition:all .2s ease; }
.stRadio label span, .stRadio label p { color:#b8acc8!important; }
.stRadio label:hover { border-color:rgba(232,121,249,.48); background:rgba(168,85,247,.09); }
.stRadio label:has(input:checked) { border-color:#a855f7; background:rgba(168,85,247,.16); box-shadow:0 0 16px rgba(168,85,247,.12); }
[data-testid="stSlider"] [role="slider"] { background:#a855f7!important; border-color:#e879f9!important; box-shadow:0 0 0 2px rgba(168,85,247,.16)!important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div { background:#a855f7!important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div > div { background:#a855f7!important; border-color:#e879f9!important; }
.stSlider [data-testid="stMarkdownContainer"] { color:#c084fc!important; }
.stRadio [data-baseweb="radio"] [aria-checked="true"] > div:first-child { background:#a855f7!important; border-color:#e879f9!important; }
.stRadio [data-baseweb="radio"] [aria-checked="true"] svg { fill:#a855f7!important; color:#a855f7!important; }
.empty-state { margin:1.35rem 0; padding:1.55rem 1.25rem; text-align:center; border:1px dashed rgba(168,85,247,.3); border-radius:16px; background:rgba(16,11,27,.45); }
.empty-state-icon { width:12px; height:12px; margin:0 auto; border-radius:50%; background:#e879f9; box-shadow:0 0 16px rgba(232,121,249,.7); }
.empty-state-title { color:#f7f3ff; font-size:1.05rem; font-weight:750; margin:.55rem 0 .3rem; }
.empty-state-copy { color:#8f849e; font-size:.8rem; line-height:1.6; max-width:560px; margin:0 auto; }
.analysis-status { display:inline-flex; align-items:center; gap:.4rem; margin-top:.65rem; padding:.35rem .65rem; border:1px solid rgba(168,85,247,.25); border-radius:999px; color:#b8acc8; font:600 .65rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; }
.analysis-status::before { content:""; width:6px; height:6px; border-radius:50%; background:#6ee7b7; box-shadow:0 0 10px #6ee7b7; }
.workflow-strip { display:flex; align-items:center; gap:.55rem; margin:0 0 1.25rem; padding:.65rem .8rem; border:1px solid rgba(168,85,247,.17); border-radius:12px; background:rgba(16,11,27,.48); }
.workflow-step { display:flex; align-items:center; gap:.42rem; color:#a89bb8; font:600 .65rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; white-space:nowrap; }
.workflow-step strong { display:grid; place-items:center; width:21px; height:21px; border-radius:50%; color:#f7f3ff; background:linear-gradient(135deg,#a855f7,#7c3aed); box-shadow:0 0 13px rgba(168,85,247,.28); }
.workflow-arrow { color:#6f627f; font-size:.75rem; }
.evidence-heading { display:flex; align-items:center; gap:.55rem; margin:1rem 0 .5rem; color:#f7f3ff; font-size:.85rem; font-weight:750; }
.evidence-heading::before { content:""; width:5px; height:18px; border-radius:999px; background:#a855f7; box-shadow:0 0 12px rgba(168,85,247,.45); }
.evidence-heading.partial::before { background:#fbbf24; box-shadow:0 0 12px rgba(251,191,36,.36); }
.evidence-heading.gaps::before { background:#f472b6; box-shadow:0 0 12px rgba(244,114,182,.36); }
.stFileUploader section { min-height:108px; display:grid; place-items:center; }
.stFileUploader section small { color:#8f849e!important; }
.brand-footer { margin-top:2.5rem; padding:1.1rem 0 .2rem; border-top:1px solid rgba(168,85,247,.18); display:flex; align-items:center; justify-content:space-between; gap:1rem; color:#756b86; font-size:.68rem; }
.brand-footer strong { color:#c084fc; font-weight:750; }
.brand-footer a { color:#e879f9; text-decoration:none; margin-left:.8rem; }
.brand-footer a:hover { text-decoration:underline; }
.dashboard-divider { height:1px; margin:1.5rem 0; background:linear-gradient(90deg, transparent, rgba(168,85,247,.38), transparent); }
.result-reveal { animation:resultReveal .5s ease-out both; }
.reveal-delay-1 { animation-delay:0ms; }
.reveal-delay-2 { animation-delay:100ms; }
.reveal-delay-3 { animation-delay:200ms; }
.reveal-delay-4 { animation-delay:300ms; }
.reveal-delay-5 { animation-delay:400ms; }
.skill-panel .tag-matched, .skill-panel .tag-missing { animation:resultReveal .45s ease-out both; }
.skill-panel .tag-matched:nth-child(2), .skill-panel .tag-missing:nth-child(2) { animation-delay:80ms; }
.skill-panel .tag-matched:nth-child(3), .skill-panel .tag-missing:nth-child(3) { animation-delay:160ms; }
.skill-panel .tag-matched:nth-child(4), .skill-panel .tag-missing:nth-child(4) { animation-delay:240ms; }
.skill-panel .tag-matched:nth-child(5), .skill-panel .tag-missing:nth-child(5) { animation-delay:320ms; }
.insight-note { padding:.75rem .9rem; border-left:2px solid #a855f7; border-radius:0 9px 9px 0; background:rgba(168,85,247,.06); color:#b8acc8; font-size:.78rem; line-height:1.6; }
.skill-panel { min-height:86px; padding:.9rem; border:1px solid rgba(168,85,247,.18); border-radius:12px; background:rgba(16,11,27,.55); }
.skill-panel-title { color:#f7f3ff; font-size:.78rem; font-weight:750; margin-bottom:.45rem; }
.streamlit-expanderHeader { font-weight:650!important; }
.stMarkdown, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color:#f7f3ff; }
.score-box, .batch-row {
    background:linear-gradient(145deg, rgba(23,15,39,.9), rgba(13,11,22,.86));
    border:1px solid rgba(168,85,247,.22); border-radius:14px; padding:1.1rem; text-align:center; margin-bottom:.8rem;
    box-shadow:inset 0 1px rgba(255,255,255,.035), 0 12px 30px rgba(0,0,0,.14); transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.score-box:hover, .batch-row:hover { transform:translateY(-3px); border-color:rgba(232,121,249,.55); box-shadow:0 14px 36px rgba(124,58,237,.18); }
.score-box {
    min-height:108px; display:flex; flex-direction:column; justify-content:center;
}
.score-number { font-size:2.2rem; font-weight:800; color:#e879f9; text-shadow:0 0 22px rgba(232,121,249,.25); }
.score-label  { font-size:.7rem; color:#b8acc8; letter-spacing:.1em; text-transform:uppercase; }
.score-sub    { font-size:.7rem; color:#756b86; margin-top:2px; }
.verdict-box  { border-radius:12px; padding:.75rem 1.1rem; font-size:.88rem; font-weight:600; margin-bottom:1rem; box-shadow:0 0 24px rgba(124,58,237,.10); }
.tag-matched  { background:rgba(74,222,170,.10); border:1px solid rgba(74,222,170,.35); color:#6ee7b7;
                border-radius:20px; padding:3px 11px; font-size:.76rem; display:inline-block; margin:3px; }
.tag-missing  { background:rgba(244,114,182,.10); border:1px solid rgba(244,114,182,.36); color:#f9a8d4;
                border-radius:20px; padding:3px 11px; font-size:.76rem; display:inline-block; margin:3px; }
.tag-semantic  { background:rgba(168,85,247,.12); border:1px solid rgba(168,85,247,.38); color:#d8b4fe;
                border-radius:20px; padding:3px 11px; font-size:.76rem; display:inline-block; margin:3px; }
.batch-title  { font-size:.88rem; font-weight:600; color:#f7f3ff; margin-bottom:.4rem; }
.score-bar { width:100%; height:4px; margin:.2rem 0 .75rem; overflow:hidden; border-radius:999px; background:rgba(168,85,247,.12); }
.score-bar span { display:block; width:100%; max-width:100%; min-width:0; height:100%; border-radius:inherit; box-shadow:none; }
.stTextArea textarea, .stTextInput input, [data-baseweb="input"] input { background:rgba(13,10,23,.88)!important; color:#f7f3ff!important; border:1px solid rgba(168,85,247,.25)!important; border-radius:11px!important; font-size:.85rem!important; transition:border-color .2s ease, box-shadow .2s ease; }
.stTextArea [data-baseweb="base-input"], .stTextInput [data-baseweb="base-input"], [data-baseweb="input"] { background:rgba(13,10,23,.88)!important; border-color:rgba(168,85,247,.25)!important; }
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color:#b8acc8!important; opacity:1!important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color:#e879f9!important; box-shadow:0 0 0 1px #e879f9, 0 0 22px rgba(232,121,249,.12)!important; }
.stButton>button, .stDownloadButton>button { background:linear-gradient(135deg, #a855f7, #7c3aed)!important; color:#fff!important; border:1px solid rgba(232,121,249,.45)!important; border-radius:10px!important; padding:.62rem 1.5rem!important; font-weight:750!important; font-size:.95rem!important; width:100%!important; margin:.15rem 0!important; box-shadow:0 8px 22px rgba(124,58,237,.22); transition:transform .2s ease, box-shadow .2s ease, filter .2s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { filter:brightness(1.12); transform:translateY(-2px); box-shadow:0 12px 28px rgba(168,85,247,.34); }
.stButton>button:active, .stDownloadButton>button:active { transform:translateY(0); }
.stTabs [data-baseweb="tab-list"] { background:rgba(19,12,31,.82); border:1px solid rgba(168,85,247,.2); border-radius:12px; padding:5px!important; gap:7px!important; }
.stTabs [data-baseweb="tab"] { color:#a89bb8; border-radius:9px; padding:.64rem 1.05rem!important; margin:0!important; white-space:nowrap; transition:color .2s ease, background .2s ease; }
.stTabs [data-baseweb="tab"]::before { content:""; display:inline-block; width:14px; height:14px; margin-right:.48rem; vertical-align:-2px; background-repeat:no-repeat; background-position:center; background-size:contain; opacity:.9; }
.stTabs [data-baseweb="tab"]:nth-child(1)::before { background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23e879f9' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='10.8' cy='10.8' r='6.8'/%3E%3Cpath d='m16 16 5 5'/%3E%3C/svg%3E"); }
.stTabs [data-baseweb="tab"]:nth-child(2)::before { background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 19V5'/%3E%3Cpath d='M4 19h17'/%3E%3Cpath d='M8 16v-5'/%3E%3Cpath d='M13 16V7'/%3E%3Cpath d='M18 16v-8'/%3E%3C/svg%3E"); }
.stTabs [data-baseweb="tab"]:nth-child(3)::before { background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23e879f9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 7.5 5 4h14l2 3.5v11A1.5 1.5 0 0 1 19.5 20h-15A1.5 1.5 0 0 1 3 18.5z'/%3E%3Cpath d='M3 8h18'/%3E%3C/svg%3E"); }
.stTabs [data-baseweb="tab"]:hover { color:#f7f3ff; background:rgba(168,85,247,.10); }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg, #a855f7, #7c3aed)!important; color:#fff!important; box-shadow:0 5px 16px rgba(124,58,237,.22); }
div[data-testid="stExpander"] { background:rgba(16,11,27,.66); border:1px solid rgba(168,85,247,.22)!important; border-radius:12px!important; overflow:hidden; }
div[data-testid="stExpander"] summary:hover { color:#e879f9!important; }
.stFileUploader { background:rgba(16,11,27,.7)!important; border:1px dashed rgba(168,85,247,.44)!important; border-radius:12px!important; transition:border-color .2s ease, background .2s ease; }
.stFileUploader:hover { border-color:#e879f9!important; background:rgba(30,16,48,.78)!important; }
[data-testid="stSidebar"] { width:300px!important; background:linear-gradient(180deg, #0d0915, #09070e)!important; border-right:1px solid rgba(168,85,247,.18); }
[data-testid="stSidebar"] > div { width:300px!important; }
[data-testid="stSidebar"] hr { margin:.85rem 0; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:#b8acc8; }
[data-testid="stSidebar"] table { border:1px solid rgba(168,85,247,.3)!important; }
[data-testid="stSidebar"] th { color:#e879f9!important; background:rgba(168,85,247,.10)!important; border-bottom:1px solid rgba(168,85,247,.3)!important; }
[data-testid="stSidebar"] td { color:#c7bdd2!important; border-color:rgba(168,85,247,.18)!important; }
[data-testid="stSidebar"] tr:hover td { background:rgba(168,85,247,.08)!important; }
[data-testid="stDataFrame"] { border:1px solid rgba(168,85,247,.2); border-radius:12px; overflow:hidden; box-shadow:0 10px 28px rgba(0,0,0,.18); }
[data-testid="stDataFrame"] ::-webkit-scrollbar-corner, [data-testid="stDataFrame"]::-webkit-scrollbar-corner { background:transparent!important; }
.evidence-table-wrap { overflow-x:auto; border:1px solid rgba(168,85,247,.18); border-radius:12px; background:rgba(12,8,20,.58); }
.evidence-table, .leaderboard-table { width:100%; border-collapse:collapse; color:#c7bdd2; font-size:.78rem; }
.evidence-table th, .evidence-table td { padding:.7rem .75rem; border-bottom:1px solid rgba(168,85,247,.12); text-align:left; vertical-align:top; white-space:normal; overflow-wrap:anywhere; }
.evidence-table th:nth-child(3), .evidence-table td:nth-child(3) { min-width:110px; word-break:normal; overflow-wrap:normal; }
.evidence-table th { color:#e879f9; background:rgba(168,85,247,.09); font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; }
.evidence-table tr:last-child td { border-bottom:0; }
.evidence-score-cell { min-width:120px; }
.evidence-score-label { font-weight:750; white-space:nowrap; }
.evidence-score-bar, .leaderboard-score-bar { height:4px; margin-top:.35rem; overflow:hidden; border-radius:999px; background:rgba(168,85,247,.12); }
.evidence-score-bar span, .leaderboard-score-bar span { display:block; height:100%; border-radius:inherit; }
.leaderboard-wrap { overflow-x:auto; border:1px solid rgba(168,85,247,.22); border-radius:14px; background:rgba(12,8,20,.58); }
.leaderboard-table th, .leaderboard-table td { padding:.72rem .8rem; border-bottom:1px solid rgba(168,85,247,.13); text-align:left; vertical-align:middle; white-space:normal; }
.leaderboard-table th { color:#e879f9; background:rgba(168,85,247,.1); font-size:.67rem; letter-spacing:.08em; text-transform:uppercase; }
.leaderboard-table tr:last-child td { border-bottom:0; }
.leaderboard-rank { color:#c084fc; font-weight:850; white-space:nowrap; }
.leaderboard-name { color:#f7f3ff; font-weight:700; min-width:180px; overflow-wrap:anywhere; }
.leaderboard-score { min-width:150px; font-weight:800; white-space:nowrap; }
.leaderboard-chip { display:inline-block; margin:2px 3px 2px 0; padding:3px 7px; border:1px solid rgba(168,85,247,.28); border-radius:999px; color:#d8b4fe; background:rgba(168,85,247,.1); font-size:.66rem; white-space:nowrap; }
.leaderboard-chip.missing { color:#f9a8d4; border-color:rgba(244,114,182,.36); background:rgba(244,114,182,.1); }
[data-testid="stPlotlyChart"] { border:1px solid rgba(168,85,247,.18); border-radius:14px; padding:.35rem; background:rgba(12,8,20,.58); animation:resultReveal .5s ease-out 300ms both; }
[data-testid="stPlotlyChart"] [data-testid="stElementToolbar"], [data-testid="stPlotlyChart"] [data-testid="stElementToolbarButton"] { display:none!important; }
.stAlert { border-radius:11px!important; border:1px solid rgba(168,85,247,.24)!important; background:rgba(25,15,40,.76)!important; }
.stSpinner { color:#b8acc8!important; }
div[data-testid="stSpinner"] { display:flex!important; align-items:center; gap:.65rem; color:#b8acc8!important; }
div[data-testid="stSpinner"] svg { display:none!important; }
div[data-testid="stSpinner"]::before { content:""; width:110px; height:2px; flex:0 0 110px; border-radius:999px; background:linear-gradient(90deg, transparent, #a855f7 35%, #e879f9 50%, transparent 65%); background-size:200% 100%; animation:spinnerSweep 1.15s linear infinite; box-shadow:0 0 12px rgba(168,85,247,.6); }
.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#b8acc8!important; }
.input-meta { display:flex; justify-content:space-between; gap:.75rem; margin:.45rem .2rem .15rem; color:#756b86; font:600 .62rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing:.03em; }
.input-meta strong { color:#c084fc; font-weight:700; }
.result-hero-note { color:#8f849e; font-size:.72rem; margin-top:.3rem; }
.stApp [data-testid="stToolbar"] { display:none!important; }
.stApp header[data-testid="stHeader"], .stApp [data-testid="stHeader"] { background:rgba(7,7,13,.96)!important; }
.stApp [data-testid="stDecoration"] { background:linear-gradient(90deg, #7c3aed, #a855f7, #e879f9)!important; }
.stTextArea { padding:.25rem; border:1px solid rgba(168,85,247,.13); border-radius:13px; background:rgba(12,8,20,.3); box-shadow:inset 0 1px rgba(255,255,255,.02); }
.stTextArea:focus-within { border-color:rgba(232,121,249,.42); box-shadow:0 0 22px rgba(168,85,247,.10), inset 0 1px rgba(255,255,255,.03); }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important; scroll-behavior:auto!important; }
    .app-hero::after, .result-gauge-reveal, div[data-testid="stSpinner"]::before, .result-reveal, .skill-panel .tag-matched, .skill-panel .tag-missing { animation:none!important; }
    .app-hero::after { --scanner-angle:0deg; }
    .result-gauge-reveal { opacity:1!important; transform:none!important; }
    .result-reveal, .skill-panel .tag-matched, .skill-panel .tag-missing { opacity:1!important; transform:none!important; }
    div[data-testid="stSpinner"]::before { background:#a855f7; }
}
@media (max-width: 768px) {
    .block-container { padding:1.3rem 1rem 3rem; }
    .app-hero { padding:1.2rem 1rem 1rem; }
    .app-hero::after { right:1rem; }
    .workflow-strip { align-items:flex-start; flex-direction:column; gap:.45rem; }
    .workflow-arrow { display:none; }
    .result-hero { align-items:flex-start; }
    .brand-footer { align-items:flex-start; flex-direction:column; }
    .brand-footer a { margin-left:0; margin-right:.8rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="sidebar-brand-kicker">Applied AI / Utility</div><div class="sidebar-brand-title">Resume <span>Screener</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Configuration</div>', unsafe_allow_html=True)
    st.text_input(
        "Gemini API key (optional, free)", type="password", key="gemini_api_key"
    )
    st.caption("If a key is provided, only the structured analysis summary is sent to Gemini, not the full resume or job description.")
    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Scoring model</div>', unsafe_allow_html=True)
    st.markdown(f"""
| Method | Weight |
|---|---|
| Semantic coverage | {SEMANTIC_WEIGHT:.0%} |
| Technical skill coverage | {SKILL_WEIGHT:.0%} |
| Keyword match | {KEYWORD_WEIGHT:.1%} |
| TF-IDF similarity | {TFIDF_WEIGHT:.1%} |
    """)
    st.caption(
        "Fallback when embeddings are disabled or unavailable: "
        f"Skill coverage {FALLBACK_SKILL_WEIGHT:.0%} + keyword {FALLBACK_KEYWORD_WEIGHT:.0%} "
        f"+ TF-IDF {FALLBACK_TFIDF_WEIGHT:.0%}."
    )
    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("- Semantic embeddings and evidence tables\n- PDF resume upload\n- Optional Gemini suggestions\n- Batch mode (1 vs many)")
    st.markdown("---")
    st.markdown("<p style='color:#756b86;font-size:.78rem;'>Python · Scikit-learn · FastEmbed ONNX · pypdf · Gemini (optional) · Streamlit</p>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<section class="app-hero">
  <div class="app-kicker">&gt; Applied AI / Resume Intelligence</div>
  <h1 class="app-title">AI Resume <span>Screener</span> <small style="font-size:.42em;color:#b8acc8;letter-spacing:.08em;vertical-align:middle;">v2</small></h1>
  <p class="app-subtitle">Compare resumes with job descriptions using semantic evidence, technical skill coverage, explainable scoring, and practical improvement guidance.</p>
  <div class="analysis-status">Analysis engine ready</div>
</section>
""", unsafe_allow_html=True)
st.divider()
st.markdown('''
<div class="workflow-strip">
  <div class="workflow-step"><strong>1</strong> Prepare inputs</div>
  <span class="workflow-arrow">→</span>
  <div class="workflow-step"><strong>2</strong> Analyze match</div>
  <span class="workflow-arrow">→</span>
  <div class="workflow-step"><strong>3</strong> Improve evidence</div>
</div>
''', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Single Analysis", "Batch Mode", "Bulk Resumes"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-kicker">Analysis workspace</div><div class="dashboard-heading">Match a candidate to a role</div><p class="dashboard-copy">Add the resume and job description below. The screener combines semantic evidence, skills, keywords, and TF-IDF into an explainable match profile.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── Resume input (text OR PDF) ────────────────────────────────────────────
    with col1:
        st.markdown('<div class="input-label">RESUME INPUT</div><div class="input-hint">Paste resume text or upload a selectable-text PDF.</div>', unsafe_allow_html=True)
        resume_input_mode = st.radio("INPUT METHOD", ["Paste text", "Upload PDF"], horizontal=True, key="r_mode")

        resume_text = ""
        if resume_input_mode == "Upload PDF":
            pdf_file = st.file_uploader(f"Upload resume PDF (max {MAX_UPLOAD_MB} MB)", type=["pdf"], key="resume_pdf")
            if pdf_file:
                resume_text, pages = _read_pdf(pdf_file, "Resume")
                if resume_text:
                    st.success(f"✅ Extracted {len(resume_text.split())} words from {pages} page(s)")
                    with st.expander("Preview extracted text"):
                        st.text(resume_text[:800] + ("..." if len(resume_text) > 800 else ""))
        else:
            resume_text = st.text_area("", height=280,
                placeholder="Paste your resume text here...", key="resume_ta",
                label_visibility="collapsed")
            resume_text = _limit_text(resume_text, "Resume")
        st.markdown(f'<div class="input-meta"><span><strong>{len(resume_text.split())}</strong> words</span><span>{len(resume_text):,} characters</span></div>', unsafe_allow_html=True)

    # ── JD input ──────────────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="input-label">TARGET ROLE</div><div class="input-hint">Paste the role requirements you want to evaluate against.</div>', unsafe_allow_html=True)
        _normalize_jd_widget_state("jd_ta")
        jd_text = st.text_area("", height=280,
            placeholder="Paste the job description here...", key="jd_ta",
            label_visibility="collapsed", on_change=_clean_jd_widget_state,
            args=("jd_ta",))
        jd_text = _limit_text(clean_jd_text(jd_text), "Job description")
        st.markdown(f'<div class="input-meta"><span><strong>{len(jd_text.split())}</strong> words</span><span>{len(jd_text):,} characters</span></div>', unsafe_allow_html=True)

    analyze_btn = st.button("Run Match Analysis", key="analyze_single", type="primary", use_container_width=True)
    st.caption("First run after idle can take 1-2 minutes while the model loads. Click once and wait.")

    if not analyze_btn:
        st.markdown('<div class="empty-state"><div class="empty-state-icon"></div><div class="empty-state-title">Your match workspace is ready</div><p class="empty-state-copy">Add a resume and a target job description above to generate semantic evidence, skill coverage, score breakdowns, and an actionable improvement plan.</p></div>', unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if analyze_btn:
        if not resume_text.strip() or not jd_text.strip():
            st.error("Please provide both resume and job description.")
        else:
            with st.spinner("Running semantic, TF-IDF, and keyword analysis..."):
                analyzer = ResumeAnalyzer()
                results  = analyzer.analyze(resume_text, jd_text)

            st.divider()
            st.markdown('<div class="section-kicker">Match intelligence</div><div class="dashboard-heading">Resume-to-role fit</div><p class="dashboard-copy">A weighted view of how strongly the resume supports this job description.</p>', unsafe_allow_html=True)

            tfidf_pct   = int(results["tfidf_score"]   * 100)
            kw_pct      = int(results["keyword_score"] * 100)
            semantic_pct = int(results["semantic_score"] * 100)
            skill_pct = int(results["skill_coverage"] * 100)
            overall_pct = int(results["overall_score"] * 100)

            def sc(p):
                if p >= 65: return "#6ee7b7"
                if p >= 40: return "#fbbf24"
                return "#f472b6"

            c1, c2, c3, c4, c5 = st.columns(5)
            for tile_index, (col, label, val, sub) in enumerate([
                (c1, "TF-IDF Similarity",   tfidf_pct,   "term importance"),
                (c2, "Keyword Match",        kw_pct,      "ATS simulation"),
                (c3, "Semantic Coverage",    semantic_pct, "requirement-level match"),
                (c4, "Skill Coverage",       skill_pct, "technical skills present"),
                (c5, "Overall Score",        overall_pct, "weighted average"),
            ], start=1):
                with col:
                    st.markdown(f"""<div class='score-box result-reveal reveal-delay-{tile_index}'>
                        <div class='score-number' style='color:{sc(val)}'>{val}%</div>
                        <div class='score-label'>{label}</div>
                        <div class='score-sub'>{sub}</div>
                    </div>""", unsafe_allow_html=True)

            if overall_pct >= 65:
                match_label = "Strong role alignment"
                tier_label = "Strong Match"
                tier_class = "tier-strong"
                tier_color = "#6ee7b7"
            elif overall_pct >= 40:
                match_label = "Moderate role alignment"
                tier_label = "Moderate Match"
                tier_class = "tier-moderate"
                tier_color = "#fbbf24"
            else:
                match_label = "Role alignment needs work"
                tier_label = "Weak Match"
                tier_class = "tier-weak"
                tier_color = "#f472b6"

            st.markdown(f"""<div class='result-hero {tier_class} result-reveal reveal-delay-2'>
                <div><div class='result-hero-label'>Decision signal</div><div class='result-hero-title'>{tier_label} <span style='color:var(--tier-color);font-size:.78rem;font-weight:600;'>· {match_label}</span></div><div class='result-hero-note'>Weighted fit across semantic evidence, skills, keywords, and TF-IDF.</div><div class='analysis-status'>Evidence profile generated</div></div>
                <div class='result-gauge result-gauge-reveal' style='background:conic-gradient({tier_color} {overall_pct * 3.6}deg, rgba(168,85,247,.13) 0);'><div class='result-gauge-value'>{overall_pct}%</div></div>
            </div>""", unsafe_allow_html=True)

            if not results["embeddings_used"]:
                st.info(results["embedding_message"])

            st.markdown('<div class="section-kicker result-reveal reveal-delay-3">Evidence layer</div><div class="dashboard-heading result-reveal reveal-delay-3">Why this score</div><p class="dashboard-copy result-reveal reveal-delay-3">Trace the strongest evidence, partial matches, and gaps behind the result.</p>', unsafe_allow_html=True)
            if results["embeddings_used"]:
                _render_match_group("Strong", results["strong_matches"], "No strong matches.")
                _render_match_group("Partial", results["partial_matches"], "No partial matches.")
                _render_match_group("Gaps", results["gaps"], "No gaps detected.")
                if results["general_expectations"]:
                    with st.expander("General expectations (not scored as gaps)"):
                        tips = (
                            "Add a brief example that shows how you meet this expectation.",
                            "Mention one relevant action and its outcome.",
                            "Give a concise example from a project or role.",
                        )
                        for index, item in enumerate(results["general_expectations"]):
                            st.markdown(
                                f"- **{html.escape(item['requirement'])}**: "
                                f"{tips[index % len(tips)]}"
                            )
            else:
                st.info("Requirement-level evidence is unavailable in fallback mode.")

            if results["ignored_lines"]:
                with st.expander(f"Ignored lines ({len(results['ignored_lines'])})"):
                    for ignored in results["ignored_lines"]:
                        st.text(f"{ignored['line']} · {ignored['reason']}")

            # Charts
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(
                    plot_score_breakdown(tfidf_pct, kw_pct, semantic_pct, overall_pct),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with ch2:
                st.plotly_chart(
                    plot_skill_chart(results["matched_skills"], results["missing_skills"]),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # Skill gap
            st.markdown('<div class="section-kicker result-reveal reveal-delay-4">Capability map</div><div class="dashboard-heading result-reveal reveal-delay-4">Skill gap analysis</div><p class="dashboard-copy result-reveal reveal-delay-4">See which technical skills are already supported and which ones need stronger evidence.</p>', unsafe_allow_html=True)
            sk1, sk2 = st.columns(2)
            with sk1:
                tags = " ".join([f"<span class='tag-matched'>{html.escape(s)}</span>" for s in results["matched_skills"]])
                matched_content = tags or "<span class='tag-missing'>None detected</span>"
                st.markdown(f'<div class="skill-panel result-reveal reveal-delay-5"><div class="skill-panel-title">Matched skills</div>{matched_content}</div>', unsafe_allow_html=True)
            with sk2:
                tags = " ".join([f"<span class='tag-missing'>{html.escape(s)}</span>" for s in results["missing_skills"]])
                missing_content = tags or "<span class='tag-matched'>All covered!</span>"
                st.markdown(f'<div class="skill-panel result-reveal reveal-delay-5"><div class="skill-panel-title">Missing skills</div>{missing_content}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-kicker">Action plan</div><div class="dashboard-heading">Improvement suggestions</div><p class="dashboard-copy">Turn the detected gaps into concrete resume improvements.</p>', unsafe_allow_html=True)
            with st.spinner("Preparing improvement suggestions..."):
                suggestion_data, suggestion_notice, suggestion_source = suggestions_for_analysis(
                    results, st.session_state.get("gemini_api_key") or GEMINI_API_KEY
                )
            st.caption(f"Suggestions: {suggestion_source}")
            if suggestion_notice:
                st.info(suggestion_notice)
            section_titles = {
                "missing_skills": "Missing Skills",
                "partial_matches": "Partial Matches",
                "gaps": "Gaps",
                "ats_tips": "ATS Tips",
            }
            for section, items in suggestion_data.items():
                if not items:
                    continue
                title = section_titles.get(section, section.replace("_", " ").title())
                title = " ".join(
                    word.upper() if word.upper() in {"ATS", "API", "SQL", "AWS"} else word
                    for word in title.split()
                )
                st.markdown(f"**{html.escape(title)}**")
                for index, item in enumerate(items):
                    if section == "missing_skills":
                        content = f"**{html.escape(item['skill'])}:** {html.escape(item['tip'])}"
                    elif section == "partial_matches":
                        verbs = ("strengthen", "clarify", "make more specific")
                        requirement = html.escape(item["requirement"].rstrip("."))
                        content = (
                            f"**{requirement}**; {verbs[index % len(verbs)]} “{html.escape(item['evidence'])}” "
                            f"by adding {html.escape(item['suggestion'])}"
                        )
                    elif section == "gaps":
                        requirement = html.escape(item["requirement"].strip().rstrip("."))
                        content = f"**{requirement}:** {html.escape(item['suggestion'])}"
                    else:
                        content = html.escape(item)
                    st.markdown(f"- {content}")

            # Raw data
            with st.expander("Raw scores & token stats"):
                st.json({
                    "tfidf_score":        round(results["tfidf_score"], 4),
                    "keyword_score":      round(results["keyword_score"], 4),
                    "semantic_score":     round(results["semantic_score"], 4),
                    "skill_coverage":     round(results["skill_coverage"], 4),
                    "overall_score":      round(results["overall_score"], 4),
                    "resume_tokens":      results["resume_token_count"],
                    "filtered_jd_tokens": results["jd_token_count"],
                    "raw_jd_tokens":      results["raw_jd_token_count"],
                    "filtered_requirements": results["requirement_count"],
                    "ignored_lines":      len(results["ignored_lines"]),
                    "raw_match_scores":   results["raw_match_scores"],
                    "matched_skills":     len(results["matched_skills"]),
                    "missing_skills":     len(results["missing_skills"]),
                    "required_technical_skills": results["required_technical_skills"],
                    "missing_technical_skills": results["missing_technical_skills"],
                    "embeddings_used":    results["embeddings_used"],
                    "scoring_mode":       results["scoring_mode"],
                })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Mode (1 resume vs multiple JDs)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Mode — Compare Your Resume Against Multiple Jobs")
    st.markdown("<p style='color:#a89bb8;font-size:.85rem;'>Paste or upload your resume once, then add multiple job descriptions to rank which roles suit you best.</p>", unsafe_allow_html=True)

    # Resume input
    st.markdown("**Your Resume (used for all comparisons)**")
    batch_resume_mode = st.radio("Input method", ["Paste text", "Upload PDF"], horizontal=True, key="b_mode")

    batch_resume = ""
    if batch_resume_mode == "Upload PDF":
        b_pdf = st.file_uploader(f"Upload resume PDF (max {MAX_UPLOAD_MB} MB)", type=["pdf"], key="batch_pdf")
        if b_pdf:
            batch_resume, pages = _read_pdf(b_pdf, "Resume")
            if batch_resume:
                st.success(f"✅ {len(batch_resume.split())} words extracted from {pages} page(s)")
    else:
        batch_resume = st.text_area("", height=180,
            placeholder="Paste your resume text here...", key="batch_resume_ta",
            label_visibility="collapsed")
        batch_resume = _limit_text(batch_resume, "Resume")

    st.markdown("---")

    # JD inputs
    st.markdown("**Job Descriptions (add up to 6)**")
    num_jds = st.slider("Number of job descriptions", min_value=2,
                        max_value=MAX_BATCH_JDS, value=min(3, MAX_BATCH_JDS))

    jd_entries = []
    for i in range(num_jds):
        with st.expander(f"Job {i+1}", expanded=(i < 2)):
            title = st.text_input(f"Job title / company", key=f"jd_title_{i}",
                                  placeholder=f"e.g. ML Engineer @ Google")
            _normalize_jd_widget_state(f"jd_text_{i}")
            text  = st.text_area("Job description", height=150, key=f"jd_text_{i}",
                                  placeholder="Paste job description here...",
                                  label_visibility="collapsed", on_change=_clean_jd_widget_state,
                                  args=(f"jd_text_{i}",))
            text = _limit_text(clean_jd_text(text), f"Job {i + 1} description")
            jd_entries.append({"title": title or f"Job {i+1}", "text": text})

    batch_btn = st.button("Rank Job Matches", key="batch_run", type="primary", use_container_width=True)
    st.caption("First run after idle can take 1-2 minutes while the model loads. Click once and wait.")

    if batch_btn:
        filled = [(e["title"], e["text"]) for e in jd_entries if e["text"].strip()]
        if not batch_resume.strip():
            st.error("Please provide your resume first.")
        elif len(filled) < 2:
            st.error("Please fill in at least 2 job descriptions.")
        else:
            batch_results = []
            analyzer = ResumeAnalyzer()

            progress = st.progress(0, text="Analyzing...")
            for idx, (title, jd_text) in enumerate(filled):
                with st.spinner(f"Analyzing: {title}..."):
                    r = analyzer.analyze(batch_resume, jd_text)
                    batch_results.append({
                        "title":   title,
                        "overall": int(r["overall_score"] * 100),
                        "_overall_score": r["overall_score"],
                        "tfidf":   int(r["tfidf_score"]   * 100),
                        "kw":      int(r["keyword_score"] * 100),
                        "semantic": int(r["semantic_score"] * 100),
                        "_semantic_score": r["semantic_score"],
                        "skill_coverage": int(r["skill_coverage"] * 100),
                        "_skill_coverage_score": r["skill_coverage"],
                        "matched": len(r["matched_skills"]),
                        "missing": len(r["missing_skills"]),
                        "matched_list": r["matched_skills"],
                        "missing_list": r["missing_skills"],
                    })
                progress.progress((idx + 1) / len(filled), text=f"Analyzed {idx+1}/{len(filled)}")

            progress.empty()

            # Sort by precise score, then skill coverage, then semantic coverage.
            batch_results = rank_batch_results(batch_results)

            st.divider()
            st.markdown("### Ranking — Best Matched Roles")

            # Bar chart
            st.plotly_chart(
                plot_batch_chart(batch_results),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Detailed cards
            for rank, br in enumerate(batch_results):
                medal = f"#{rank+1}"
                oc    = "#6ee7b7" if br["overall"] >= 65 else "#fbbf24" if br["overall"] >= 40 else "#f472b6"
                verdict = "Strong Match" if br["overall"] >= 65 else "Moderate" if br["overall"] >= 40 else "Weak Match"

                st.markdown(f"""<div class='batch-row'>
                    <div class='batch-title'>{medal} {html.escape(br['title'])} &nbsp;
                        <span style='color:{oc};font-size:1.1rem;font-weight:800;'>{br['overall']}%</span>
                        <span style='color:{oc};font-size:.75rem;margin-left:6px;'>{verdict}</span>
                    </div>
                    <div class='score-bar'><span style='width:{br['overall']}%;background:{oc};'></span></div>
                    <span style='font-size:.75rem;color:#a89bb8;'>
                        TF-IDF: <b style='color:#f0eeff'>{br['tfidf']}%</b> &nbsp;|&nbsp;
                        Keywords: <b style='color:#f0eeff'>{br['kw']}%</b> &nbsp;|&nbsp;
                        Semantic: <b style='color:#f0eeff'>{br['semantic']}%</b> &nbsp;|&nbsp;
                        Skill coverage: <b style='color:#f0eeff'>{br['skill_coverage']}%</b> &nbsp;|&nbsp;
                        Matched skills: <b style='color:#6ee7b7'>{br['matched']}</b> &nbsp;|&nbsp;
                        Missing: <b style='color:#f472b6'>{br['missing']}</b>
                    </span>
                </div>""", unsafe_allow_html=True)

                with st.expander(f"Skills for {br['title']}", expanded=False):
                    st.markdown("**Matched Skills**")
                    matched_tags = " ".join(
                        f"<span class='tag-matched'>{html.escape(str(skill))}</span>"
                        for skill in br["matched_list"]
                    )
                    st.markdown(matched_tags or "<span class='tag-missing'>None</span>", unsafe_allow_html=True)
                    st.markdown("**Missing Skills**")
                    missing_tags = " ".join(
                        f"<span class='tag-missing'>{html.escape(str(skill))}</span>"
                        for skill in br["missing_list"]
                    )
                    st.markdown(missing_tags or "<span class='tag-matched'>All covered!</span>", unsafe_allow_html=True)

            # Best match deep-dive
            best = batch_results[0]
            st.markdown(f"### Best Match Deep-Dive: {best['title']}")
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("**Matched Skills**")
                tags = " ".join([f"<span class='tag-matched'>{html.escape(s)}</span>" for s in best["matched_list"]])
                st.markdown(tags or "<span class='tag-missing'>None</span>", unsafe_allow_html=True)
            with bc2:
                st.markdown("**Missing Skills**")
                tags = " ".join([f"<span class='tag-missing'>{html.escape(s)}</span>" for s in best["missing_list"]])
                st.markdown(tags or "<span class='tag-matched'>All covered!</span>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Bulk Resumes (many resumes vs one JD)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Bulk Resumes - Compare Multiple Resumes Against One Job")
    _normalize_jd_widget_state("bulk_jd_ta")
    bulk_jd_text = st.text_area(
        "Job description",
        height=220,
        key="bulk_jd_ta",
        placeholder="Paste the job description here...",
        label_visibility="collapsed",
        on_change=_clean_jd_widget_state,
        args=("bulk_jd_ta",),
    )
    bulk_jd_text = _limit_text(clean_jd_text(bulk_jd_text), "Job description")
    bulk_files = st.file_uploader(
        "Resume PDFs (up to 10 files, 5 MB each)",
        type=["pdf"],
        accept_multiple_files=True,
        key="bulk_resume_pdfs",
    )

    upload_indices, upload_messages = check_upload_limits(
        [(file.name, file.size) for file in bulk_files]
    )
    for upload_message in upload_messages:
        st.warning(html.escape(upload_message))

    bulk_signature = (
        bulk_jd_text,
        tuple((file.name, file.size) for file in bulk_files),
    )
    if st.session_state.get("bulk_signature") != bulk_signature:
        st.session_state.pop("bulk_results", None)

    bulk_run = st.button("Analyze Resume Set", key="bulk_analyze", type="primary", use_container_width=True)
    st.caption("First run after idle can take 1-2 minutes while the model loads. Click once and wait.")
    if bulk_run:
        st.session_state.pop("bulk_results", None)
        if not bulk_jd_text.strip():
            st.error("Please provide a job description.")
        elif not bulk_files:
            st.error("Please upload at least one resume PDF.")
        elif not upload_indices:
            st.error("No uploaded resumes passed the file-size limit.")
        else:
            with st.spinner(
                f"Running semantic, TF-IDF, and keyword analysis on {len(upload_indices)} resumes..."
            ):
                analyzer = ResumeAnalyzer()
                prepared_jd = analyzer.prepare_job_description(
                    bulk_jd_text,
                    precompute_embeddings=True,
                )
                rows = []
                details = {}
                progress = st.progress(0, text="Analyzing resumes one at a time...")

                for processed_index, file_index in enumerate(upload_indices, start=1):
                    uploaded_file = bulk_files[file_index]
                    file_name = uploaded_file.name
                    escaped_name = html.escape(file_name)
                    analysis = None
                    resume_text = ""
                    try:
                        memory_mb = get_available_memory_mb()
                        logger.info(
                            "Available memory before bulk resume %s: %s",
                            file_name,
                            f"{memory_mb:.1f} MB" if memory_mb is not None else "unknown",
                        )
                        resume_text, _ = _read_pdf(uploaded_file, f"Resume {escaped_name}")
                        if resume_text:
                            analysis = analyzer.analyze(
                                resume_text,
                                bulk_jd_text,
                                prepared_job_description=prepared_jd,
                            )
                            row = {
                                "_detail_id": file_index,
                                "file_name": file_name,
                                "overall_pct": round(analysis["overall_score"] * 100),
                                "_overall_score": analysis["overall_score"],
                                "semantic_pct": round(analysis["semantic_score"] * 100),
                                "_semantic_score": analysis["semantic_score"],
                                "skill_coverage_pct": round(analysis["skill_coverage"] * 100),
                                "_skill_coverage_score": analysis["skill_coverage"],
                                "strong_count": len(analysis["strong_matches"]),
                                "partial_count": len(analysis["partial_matches"]),
                                "gaps_count": len(analysis["gaps"]),
                                "top_missing_skills": ", ".join(analysis["missing_skills"][:3]),
                                "mode": "semantic" if analysis["embeddings_used"] else "fallback",
                            }
                            rows.append(row)
                            details[file_index] = {
                                "mode": row["mode"],
                                "embedding_message": analysis["embedding_message"],
                                "strong_matches": analysis["strong_matches"],
                                "partial_matches": analysis["partial_matches"],
                                "gaps": analysis["gaps"],
                            }
                    except Exception:
                        logger.exception("Bulk resume analysis failed for %s", file_name)
                        st.error(f"Could not process {escaped_name}; the other resumes will continue.")
                    finally:
                        memory_mb = get_available_memory_mb()
                        logger.info(
                            "Available memory after bulk resume %s: %s",
                            file_name,
                            f"{memory_mb:.1f} MB" if memory_mb is not None else "unknown",
                        )
                        del analysis, resume_text, uploaded_file
                        gc.collect()
                        progress.progress(
                            processed_index / len(upload_indices),
                            text=f"Processed {processed_index} of {len(upload_indices)} files",
                        )

                progress.empty()
                ranked_rows = rank_results(rows)
                details_by_rank = {}
                for row in ranked_rows:
                    details_by_rank[row["rank"]] = details.pop(row.pop("_detail_id"))
                st.session_state["bulk_signature"] = bulk_signature
                st.session_state["bulk_results"] = {
                    "rows": ranked_rows,
                    "details": details_by_rank,
                }
                del prepared_jd, analyzer, bulk_files
                gc.collect()

    bulk_results = st.session_state.get("bulk_results")
    if bulk_results and bulk_results.get("rows"):
        ranked_rows = bulk_results["rows"]
        _render_bulk_leaderboard(ranked_rows)
        st.download_button(
            "Download ranked results (CSV)",
            data=build_ranked_csv(ranked_rows),
            file_name="bulk_resume_results.csv",
            mime="text/csv",
        )
        selected_rank = st.selectbox(
            "Resume evidence details",
            options=[row["rank"] for row in ranked_rows],
            format_func=lambda rank: (
                f"{rank}. "
                f"{next(row['file_name'] for row in ranked_rows if row['rank'] == rank)}"
            ),
        )
        selected_row = next(row for row in ranked_rows if row["rank"] == selected_rank)
        selected_details = bulk_results["details"][selected_rank]
        st.markdown(f"### Evidence: {html.escape(selected_row['file_name'])}")
        if selected_details["mode"] == "fallback":
            st.info(selected_details["embedding_message"] or "Requirement evidence is unavailable in fallback mode.")
        else:
            _render_match_group("Strong", selected_details["strong_matches"], "No strong matches.")
            _render_match_group("Partial", selected_details["partial_matches"], "No partial matches.")
            _render_match_group("Gaps", selected_details["gaps"], "No gaps detected.")
    elif bulk_results:
        st.info("No resumes could be analyzed. Review the file messages above and try again.")

st.markdown('''
<footer class="brand-footer">
  <div><strong>AI Resume Screener</strong> · built by Ashutosh Gupta</div>
  <div><a href="https://github.com/ashutosh8950/ai-resume-screener" target="_blank" rel="noopener noreferrer">GitHub</a><a href="https://www.linkedin.com/in/im-ashutosh-gupta/" target="_blank" rel="noopener noreferrer">LinkedIn</a></div>
</footer>
''', unsafe_allow_html=True)
