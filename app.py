"""
app.py  (v2 — upgraded)
Streamlit UI with three new features:
    1. PDF resume upload (pypdf)
    2. Pretrained semantic similarity score (FastEmbed ONNX)
    3. Batch mode — 1 resume vs multiple JDs
"""

import html

import pandas as pd
import streamlit as st

from resume_screener.analyzer import ResumeAnalyzer
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
from resume_screener.pdf_parser import PDFParseError, PDFParser
from resume_screener.suggestions import suggestions_for_analysis
from resume_screener.visualizer import plot_batch_chart, plot_score_breakdown, plot_skill_chart


def _limit_text(text: str, label: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        st.warning(f"{label} exceeded {MAX_TEXT_CHARS:,} characters and was truncated.")
        return text[:MAX_TEXT_CHARS]
    return text


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
    st.markdown(f"**{title} ({len(pairs)})**")
    if not pairs:
        st.caption(empty_message)
        return

    def render_table(rows: list[dict]) -> None:
        rendered_rows = []
        for pair in rows:
            note_parts = [pair.get("status_reason", ""), f"cosine {pair['raw_score']:.2f}"]
            if pair.get("optional"):
                note_parts.insert(0, "Optional · weight 0.5")
            note = ". ".join(part.strip().rstrip(".") for part in note_parts if part.strip())
            rendered_rows.append(
                {
                    "Requirement": pair["requirement"],
                    "Score": f"{pair['status']} · {pair['score']:.0%}",
                    "Section": pair["section"],
                    "Best evidence": pair["evidence"] or "No eligible resume evidence",
                    "Note": note,
                }
            )
        frame = pd.DataFrame(
            rendered_rows,
            columns=["Requirement", "Score", "Section", "Best evidence", "Note"],
        )
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            height=38 + 35 * len(frame),
            column_config={
                "Requirement": st.column_config.TextColumn(width="large"),
                "Score": st.column_config.TextColumn(width=160),
                "Section": st.column_config.TextColumn(width="medium"),
                "Best evidence": st.column_config.TextColumn(width="large"),
                "Note": st.column_config.TextColumn(width="large"),
            },
        )

    render_table(pairs[:8])

    if len(pairs) > 8:
        with st.expander(f"Show {len(pairs) - 8} more {title.lower()} matches"):
            render_table(pairs[8:])

st.set_page_config(page_title="AI Resume Screener v2", page_icon="📄", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0e0e16; color: #f0eeff; }
.block-container { padding-top: 1.5rem; }
h1,h2,h3 { color: #f0eeff; }
.score-box {
    background:#13131a; border:1px solid #2a2840;
    border-radius:12px; padding:1.1rem; text-align:center; margin-bottom:.8rem;
}
.score-number { font-size:2.2rem; font-weight:800; }
.score-label  { font-size:.7rem; color:#9b96c9; letter-spacing:.1em; text-transform:uppercase; }
.score-sub    { font-size:.7rem; color:#5c5880; margin-top:2px; }
.verdict-box  { border-radius:10px; padding:.7rem 1.1rem; font-size:.88rem; font-weight:600; margin-bottom:1rem; }
.tag-matched  { background:rgba(61,223,160,.12); border:1px solid rgba(61,223,160,.35); color:#3ddfa0;
                border-radius:20px; padding:3px 11px; font-size:.76rem; display:inline-block; margin:3px; }
.tag-missing  { background:rgba(255,107,107,.12); border:1px solid rgba(255,107,107,.35); color:#ff6b6b;
                border-radius:20px; padding:3px 11px; font-size:.76rem; display:inline-block; margin:3px; }
.tag-semantic  { background:rgba(124,107,255,.12); border:1px solid rgba(124,107,255,.35); color:#b8a9ff;
                border-radius:20px; padding:3px 11px; font-size:.76rem; display:inline-block; margin:3px; }
.batch-row    { background:#13131a; border:1px solid #2a2840; border-radius:10px;
                padding:.9rem 1.1rem; margin-bottom:.6rem; }
.batch-title  { font-size:.88rem; font-weight:600; color:#f0eeff; margin-bottom:.4rem; }
.stTextArea textarea { background-color:#13131a!important; color:#f0eeff!important;
                       border:1px solid #2a2840!important; border-radius:10px!important; font-size:.85rem!important; }
.stButton>button { background:#7c6bff; color:white; border:none; border-radius:10px;
                   padding:.55rem 1.5rem; font-weight:700; font-size:.95rem; width:100%; }
.stButton>button:hover { background:#6a59f0; }
.stTabs [data-baseweb="tab-list"] { background:#13131a; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"]      { color:#9b96c9; border-radius:8px; }
.stTabs [aria-selected="true"]    { background:#7c6bff!important; color:white!important; }
div[data-testid="stExpander"]     { border:1px solid #2a2840!important; border-radius:10px!important; }
.stFileUploader { background:#13131a!important; border:1px dashed #2a2840!important; border-radius:10px!important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.text_input(
        "Gemini API key (optional, free)", type="password", key="gemini_api_key"
    )
    st.caption("If a key is provided, only the structured analysis summary is sent to Gemini, not the full resume or job description.")
    st.markdown("---")
    st.markdown("**Scoring weights:**")
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
    st.markdown("**New in v2:**")
    st.markdown("- 🧠 Semantic embeddings and evidence tables\n- 📎 PDF resume upload\n- ✨ Optional Gemini suggestions\n- 📊 Batch mode (1 vs many)")
    st.markdown("---")
    st.markdown("<p style='color:#5c5880;font-size:.78rem;'>Python · Scikit-learn · FastEmbed ONNX · pypdf · Gemini (optional) · Streamlit</p>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📄 AI Resume Screener &nbsp; <span style='font-size:.75rem;background:#23233a;color:#9b96c9;padding:3px 10px;border-radius:20px;vertical-align:middle;'>v2</span>", unsafe_allow_html=True)
st.markdown("<p style='color:#9b96c9;font-size:.88rem;margin-top:-.3rem;'>Semantic Match · Evidence Tables · TF-IDF · Keyword Match · Optional Gemini Suggestions · PDF Upload · Batch Mode</p>", unsafe_allow_html=True)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Single Analysis", "📊  Batch Mode"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    col1, col2 = st.columns(2)

    # ── Resume input (text OR PDF) ────────────────────────────────────────────
    with col1:
        st.markdown("**📋 Your Resume**")
        resume_input_mode = st.radio("Input method", ["Paste text", "Upload PDF"], horizontal=True, key="r_mode")

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

    # ── JD input ──────────────────────────────────────────────────────────────
    with col2:
        st.markdown("**💼 Job Description**")
        jd_text = st.text_area("", height=280,
            placeholder="Paste the job description here...", key="jd_ta",
            label_visibility="collapsed")
        jd_text = _limit_text(jd_text, "Job description")

    analyze_btn = st.button("🔍  Analyze Match", key="analyze_single")

    # ── Results ───────────────────────────────────────────────────────────────
    if analyze_btn:
        if not resume_text.strip() or not jd_text.strip():
            st.error("Please provide both resume and job description.")
        else:
            with st.spinner("Running semantic, TF-IDF, and keyword analysis..."):
                analyzer = ResumeAnalyzer()
                results  = analyzer.analyze(resume_text, jd_text)

            st.divider()
            st.markdown("### 📊 Match Scores")

            tfidf_pct   = int(results["tfidf_score"]   * 100)
            kw_pct      = int(results["keyword_score"] * 100)
            semantic_pct = int(results["semantic_score"] * 100)
            skill_pct = int(results["skill_coverage"] * 100)
            overall_pct = int(results["overall_score"] * 100)

            def sc(p):
                if p >= 65: return "#3ddfa0"
                if p >= 40: return "#f5c842"
                return "#ff6b6b"

            c1, c2, c3, c4, c5 = st.columns(5)
            for col, label, val, sub in [
                (c1, "TF-IDF Similarity",   tfidf_pct,   "term importance"),
                (c2, "Keyword Match",        kw_pct,      "ATS simulation"),
                (c3, "Semantic Coverage",    semantic_pct, "requirement-level match"),
                (c4, "Skill Coverage",       skill_pct, "technical skills present"),
                (c5, "Overall Score",        overall_pct, "weighted average"),
            ]:
                with col:
                    st.markdown(f"""<div class='score-box'>
                        <div class='score-number' style='color:{sc(val)}'>{val}%</div>
                        <div class='score-label'>{label}</div>
                        <div class='score-sub'>{sub}</div>
                    </div>""", unsafe_allow_html=True)

            # Verdict
            if overall_pct >= 65:
                vs = "background:rgba(61,223,160,.12);border:1px solid rgba(61,223,160,.35);color:#3ddfa0;"
                vt = "✅ Strong Match — great alignment with this role!"
            elif overall_pct >= 40:
                vs = "background:rgba(245,200,66,.12);border:1px solid rgba(245,200,66,.35);color:#f5c842;"
                vt = "⚠️ Moderate Match — a few gaps to address."
            else:
                vs = "background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.35);color:#ff6b6b;"
                vt = "❌ Weak Match — significant improvements needed."
            st.markdown(f"<div class='verdict-box' style='{vs}'>{vt}</div>", unsafe_allow_html=True)

            if not results["embeddings_used"]:
                st.info(results["embedding_message"])

            st.markdown("### Why this score")
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
                    use_container_width=True
                )
            with ch2:
                st.plotly_chart(
                    plot_skill_chart(results["matched_skills"], results["missing_skills"]),
                    use_container_width=True
                )

            # Skill gap
            st.markdown("### 🔬 Skill Gap Analysis")
            sk1, sk2 = st.columns(2)
            with sk1:
                st.markdown("**✅ Matched Skills**")
                tags = " ".join([f"<span class='tag-matched'>{html.escape(s)}</span>" for s in results["matched_skills"]])
                st.markdown(tags or "<span class='tag-missing'>None detected</span>", unsafe_allow_html=True)
            with sk2:
                st.markdown("**❌ Missing Skills**")
                tags = " ".join([f"<span class='tag-missing'>{html.escape(s)}</span>" for s in results["missing_skills"]])
                st.markdown(tags or "<span class='tag-matched'>All covered!</span>", unsafe_allow_html=True)

            st.markdown("### Improvement Suggestions")
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
                        content = f"**{html.escape(item['requirement'])}:** {html.escape(item['suggestion'])}"
                    else:
                        content = html.escape(item)
                    st.markdown(f"- {content}")

            # Raw data
            with st.expander("🔢 Raw scores & token stats"):
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
    st.markdown("### 📊 Batch Mode — Compare Your Resume Against Multiple Jobs")
    st.markdown("<p style='color:#9b96c9;font-size:.85rem;'>Paste or upload your resume once, then add multiple job descriptions to rank which roles suit you best.</p>", unsafe_allow_html=True)

    # Resume input
    st.markdown("**📋 Your Resume (used for all comparisons)**")
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
    st.markdown("**💼 Job Descriptions (add up to 6)**")
    num_jds = st.slider("Number of job descriptions", min_value=2,
                        max_value=MAX_BATCH_JDS, value=min(3, MAX_BATCH_JDS))

    jd_entries = []
    for i in range(num_jds):
        with st.expander(f"Job {i+1}", expanded=(i < 2)):
            title = st.text_input(f"Job title / company", key=f"jd_title_{i}",
                                  placeholder=f"e.g. ML Engineer @ Google")
            text  = st.text_area("Job description", height=150, key=f"jd_text_{i}",
                                  placeholder="Paste job description here...",
                                  label_visibility="collapsed")
            text = _limit_text(text, f"Job {i + 1} description")
            jd_entries.append({"title": title or f"Job {i+1}", "text": text})

    batch_btn = st.button("📊  Run Batch Analysis", key="batch_run")

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
                        "tfidf":   int(r["tfidf_score"]   * 100),
                        "kw":      int(r["keyword_score"] * 100),
                        "semantic": int(r["semantic_score"] * 100),
                        "skill_coverage": int(r["skill_coverage"] * 100),
                        "matched": len(r["matched_skills"]),
                        "missing": len(r["missing_skills"]),
                        "matched_list": r["matched_skills"],
                        "missing_list": r["missing_skills"],
                    })
                progress.progress((idx + 1) / len(filled), text=f"Analyzed {idx+1}/{len(filled)}")

            progress.empty()

            # Sort by overall score
            batch_results.sort(key=lambda x: x["overall"], reverse=True)

            st.divider()
            st.markdown("### 🏆 Ranking — Best Matched Roles")

            # Bar chart
            st.plotly_chart(
                plot_batch_chart(batch_results),
                use_container_width=True
            )

            # Detailed cards
            for rank, br in enumerate(batch_results):
                medal = ["🥇", "🥈", "🥉"][rank] if rank < 3 else f"#{rank+1}"
                oc    = "#3ddfa0" if br["overall"] >= 65 else "#f5c842" if br["overall"] >= 40 else "#ff6b6b"
                verdict = "Strong Match" if br["overall"] >= 65 else "Moderate" if br["overall"] >= 40 else "Weak Match"

                st.markdown(f"""<div class='batch-row'>
                    <div class='batch-title'>{medal} {html.escape(br['title'])} &nbsp;
                        <span style='color:{oc};font-size:1.1rem;font-weight:800;'>{br['overall']}%</span>
                        <span style='color:{oc};font-size:.75rem;margin-left:6px;'>{verdict}</span>
                    </div>
                    <span style='font-size:.75rem;color:#9b96c9;'>
                        TF-IDF: <b style='color:#f0eeff'>{br['tfidf']}%</b> &nbsp;|&nbsp;
                        Keywords: <b style='color:#f0eeff'>{br['kw']}%</b> &nbsp;|&nbsp;
                        Semantic: <b style='color:#f0eeff'>{br['semantic']}%</b> &nbsp;|&nbsp;
                        Skill coverage: <b style='color:#f0eeff'>{br['skill_coverage']}%</b> &nbsp;|&nbsp;
                        Matched skills: <b style='color:#3ddfa0'>{br['matched']}</b> &nbsp;|&nbsp;
                        Missing: <b style='color:#ff6b6b'>{br['missing']}</b>
                    </span>
                </div>""", unsafe_allow_html=True)

            # Best match deep-dive
            best = batch_results[0]
            st.markdown(f"### 🔬 Best Match Deep-Dive: {best['title']}")
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("**✅ Matched Skills**")
                tags = " ".join([f"<span class='tag-matched'>{html.escape(s)}</span>" for s in best["matched_list"]])
                st.markdown(tags or "<span class='tag-missing'>None</span>", unsafe_allow_html=True)
            with bc2:
                st.markdown("**❌ Missing Skills**")
                tags = " ".join([f"<span class='tag-missing'>{html.escape(s)}</span>" for s in best["missing_list"]])
                st.markdown(tags or "<span class='tag-matched'>All covered!</span>", unsafe_allow_html=True)
