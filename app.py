"""
app.py  (v2 — upgraded)
Streamlit UI with three new features:
  1. PDF resume upload (PyPDF2)
  2. Word2Vec semantic similarity score (gensim)
  3. Batch mode — 1 resume vs multiple JDs
"""

import html

import streamlit as st

from resume_screener.analyzer import ResumeAnalyzer
from resume_screener.config import MAX_BATCH_JDS, MAX_TEXT_CHARS, MAX_UPLOAD_MB, OPENAI_API_KEY
from resume_screener.pdf_parser import PDFParseError, PDFParser
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
.tag-w2v      { background:rgba(124,107,255,.12); border:1px solid rgba(124,107,255,.35); color:#b8a9ff;
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
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...") or OPENAI_API_KEY
    st.markdown("---")
    st.markdown("**Scoring weights:**")
    st.markdown("""
| Method | Weight |
|---|---|
| TF-IDF similarity | 35% |
| Keyword match | 35% |
| Word2Vec semantic | 30% |
    """)
    st.markdown("---")
    st.markdown("**New in v2:**")
    st.markdown("- 📎 PDF resume upload\n- 🧠 Word2Vec semantic score\n- 📊 Batch mode (1 vs many)")
    st.markdown("---")
    st.markdown("<p style='color:#5c5880;font-size:.78rem;'>Python · Sklearn · Gensim · PyPDF2 · OpenAI · Streamlit</p>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📄 AI Resume Screener &nbsp; <span style='font-size:.75rem;background:#23233a;color:#9b96c9;padding:3px 10px;border-radius:20px;vertical-align:middle;'>v2</span>", unsafe_allow_html=True)
st.markdown("<p style='color:#9b96c9;font-size:.88rem;margin-top:-.3rem;'>TF-IDF · Keyword Match · Word2Vec · GPT Suggestions · PDF Upload · Batch Mode</p>", unsafe_allow_html=True)
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
            pdf_file = st.file_uploader("Upload resume PDF", type=["pdf"], key="resume_pdf")
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
            with st.spinner("Running analysis (TF-IDF + Word2Vec + Keyword)..."):
                analyzer = ResumeAnalyzer()
                results  = analyzer.analyze(resume_text, jd_text)

            st.divider()
            st.markdown("### 📊 Match Scores")

            tfidf_pct   = int(results["tfidf_score"]   * 100)
            kw_pct      = int(results["keyword_score"] * 100)
            w2v_pct     = int(results["w2v_score"]     * 100)
            overall_pct = int(results["overall_score"] * 100)

            def sc(p):
                if p >= 70: return "#3ddfa0"
                if p >= 45: return "#f5c842"
                return "#ff6b6b"

            c1, c2, c3, c4 = st.columns(4)
            for col, label, val, sub in [
                (c1, "TF-IDF Similarity",   tfidf_pct,   "term importance"),
                (c2, "Keyword Match",        kw_pct,      "ATS simulation"),
                (c3, "Word2Vec Semantic",    w2v_pct,     "meaning-level match"),
                (c4, "Overall Score",        overall_pct, "weighted average"),
            ]:
                with col:
                    st.markdown(f"""<div class='score-box'>
                        <div class='score-number' style='color:{sc(val)}'>{val}%</div>
                        <div class='score-label'>{label}</div>
                        <div class='score-sub'>{sub}</div>
                    </div>""", unsafe_allow_html=True)

            # Verdict
            if overall_pct >= 70:
                vs = "background:rgba(61,223,160,.12);border:1px solid rgba(61,223,160,.35);color:#3ddfa0;"
                vt = "✅ Strong Match — great alignment with this role!"
            elif overall_pct >= 45:
                vs = "background:rgba(245,200,66,.12);border:1px solid rgba(245,200,66,.35);color:#f5c842;"
                vt = "⚠️ Moderate Match — a few gaps to address."
            else:
                vs = "background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.35);color:#ff6b6b;"
                vt = "❌ Weak Match — significant improvements needed."
            st.markdown(f"<div class='verdict-box' style='{vs}'>{vt}</div>", unsafe_allow_html=True)

            if not results["w2v_used"]:
                st.info(results["w2v_message"])

            # Charts
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(
                    plot_score_breakdown(tfidf_pct, kw_pct, w2v_pct, overall_pct),
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

            # Word2Vec related terms
            if results["w2v_terms"]:
                st.markdown("### 🧠 Word2Vec — Semantically Related Terms")
                st.markdown("<p style='color:#9b96c9;font-size:.82rem;'>Terms the model found semantically similar in your resume/JD — shows what concepts cluster together.</p>", unsafe_allow_html=True)
                tags = " ".join([
                    f"<span class='tag-w2v'>{html.escape(w)} <span style='opacity:.6'>({s})</span></span>"
                    for w, s in results["w2v_terms"]
                ])
                st.markdown(tags, unsafe_allow_html=True)

            # GPT suggestions
            st.markdown("### 🤖 GPT Improvement Suggestions")
            if api_key:
                with st.spinner("Generating AI suggestions..."):
                    suggestions = analyzer.get_gpt_suggestions(resume_text, jd_text, results, api_key)
                st.markdown(f"""<div style='background:#13131a;border:1px solid #2a2840;border-radius:12px;
                    padding:1.2rem;font-size:.86rem;line-height:1.85;color:#c8c4f0;
                    white-space:pre-wrap;'>{html.escape(suggestions)}</div>""", unsafe_allow_html=True)
            else:
                st.info("💡 Add your OpenAI API key in the sidebar to get GPT-powered improvement suggestions.")

            # Raw data
            with st.expander("🔢 Raw scores & token stats"):
                st.json({
                    "tfidf_score":        round(results["tfidf_score"], 4),
                    "keyword_score":      round(results["keyword_score"], 4),
                    "w2v_score":          round(results["w2v_score"], 4),
                    "overall_score":      round(results["overall_score"], 4),
                    "resume_tokens":      results["resume_token_count"],
                    "jd_tokens":          results["jd_token_count"],
                    "matched_skills":     len(results["matched_skills"]),
                    "missing_skills":     len(results["missing_skills"]),
                    "w2v_available":      results["w2v_available"],
                    "w2v_status":         results["w2v_status"],
                    "scoring_method":     results["scoring_method"],
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
        b_pdf = st.file_uploader("Upload resume PDF", type=["pdf"], key="batch_pdf")
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
                        "w2v":     int(r["w2v_score"]     * 100),
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
                oc    = "#3ddfa0" if br["overall"] >= 70 else "#f5c842" if br["overall"] >= 45 else "#ff6b6b"
                verdict = "Strong Match" if br["overall"] >= 70 else "Moderate" if br["overall"] >= 45 else "Weak Match"

                st.markdown(f"""<div class='batch-row'>
                    <div class='batch-title'>{medal} {html.escape(br['title'])} &nbsp;
                        <span style='color:{oc};font-size:1.1rem;font-weight:800;'>{br['overall']}%</span>
                        <span style='color:{oc};font-size:.75rem;margin-left:6px;'>{verdict}</span>
                    </div>
                    <span style='font-size:.75rem;color:#9b96c9;'>
                        TF-IDF: <b style='color:#f0eeff'>{br['tfidf']}%</b> &nbsp;|&nbsp;
                        Keywords: <b style='color:#f0eeff'>{br['kw']}%</b> &nbsp;|&nbsp;
                        Word2Vec: <b style='color:#f0eeff'>{br['w2v']}%</b> &nbsp;|&nbsp;
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
