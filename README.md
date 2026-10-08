# 📄 AI Resume Screener v2

An advanced resume-job description matcher using pretrained **FastEmbed semantic embeddings**, **TF-IDF**, **keyword similarity**, and optional **GPT-3.5** suggestions.

---

## 🚀 What's New in v2

| Feature | Tech |
|---|---|
| **PDF Resume Upload** | `PyPDF2` — extract text from any resume PDF |
| **Semantic requirement matching** | `fastembed` ONNX Runtime on CPU with `BAAI/bge-small-en-v1.5` |
| **Batch Mode** | Compare 1 resume against up to 6 jobs at once |
| **Evidence and gaps** | Best resume sentence for each strong JD requirement, plus unmatched requirements |
| **4-score dashboard** | Semantic match + TF-IDF + Keywords + Overall |
| **Ranked job comparison chart** | Grouped Plotly bar chart |

---

## 📁 Project Structure

```
resume_screener/
│
├── app.py            ← Streamlit UI (tabs: Single + Batch)
├── resume_screener/  ← Configuration, logging, analysis, parsing, skills, charts
├── tests/            ← pytest regression tests
├── requirements.txt  ← Runtime dependencies
├── requirements-dev.txt ← Runtime dependencies plus pytest
├── .env.example      ← Environment setting names and defaults
└── README.md
```

---

## 🛠️ Setup & Run

```bash
# 1. Install runtime and development dependencies
python -m pip install -r requirements-dev.txt

# 2. Run tests
python -m pytest -q

# 3. Run locally
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
# Open http://localhost:8501

# Render start command (keep app.py in the project root)
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Python 3.11.9 is declared in `runtime.txt` and `.python-version`. Runtime settings
are read from environment variables; see `.env.example` for names and defaults.
The embedding model downloads on the first analysis and is then cached in-process.
Set `EMBEDDINGS_ENABLED=false` to use the keyword/TF-IDF fallback without loading it.

---

## 🧠 How Scoring Works

### Semantic Matching (40%)
- Splits job descriptions into requirement lines/sentences and resumes into evidence sentences/bullets
- Embeds those snippets with `BAAI/bge-small-en-v1.5` through FastEmbed/ONNX on CPU
- Matches each requirement to its most similar resume sentence using cosine similarity
- Averages each requirement's best score; evidence and below-threshold gaps are shown in the UI

### TF-IDF Similarity (30%)
- `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)`
- Cosine similarity between resume and JD vectors
- Captures term importance, not just frequency

### Keyword Match (30%)
- Tokenizes JD, removes stopwords
- Checks fraction of JD keywords present in resume
- Simulates how ATS systems scan resumes

### Overall Score
```python
# With semantic embeddings:
overall = semantic * 0.40 + keyword * 0.30 + tfidf * 0.30

# If embeddings are disabled or unavailable:
overall = keyword * 0.50 + tfidf * 0.50
```

---

## 📎 PDF Parser

```python
from pdf_parser import PDFParser

parser = PDFParser()
text   = parser.extract_text(open("resume.pdf", "rb").read())
pages  = parser.get_page_count(open("resume.pdf", "rb").read())
```

Handles:
- Multi-page PDFs
- Hyphenated line breaks
- Non-printable characters
- Excess whitespace normalization

---

## 📊 Batch Mode

Upload one resume, add up to 6 job descriptions, click **Run Batch Analysis**.

Output:
- Ranked table of jobs by match %
- Grouped bar chart (Semantic / TF-IDF / Keywords / Overall per job)
- Deep-dive skill gap for the top match

---

## 📌 Resume Bullet Points (copy-paste ready)

```
AI Resume Screener v2 | Python, Scikit-learn, FastEmbed, ONNX Runtime, PyPDF2, OpenAI, Streamlit, Plotly

• Built a resume-job matcher combining pretrained semantic embeddings, TF-IDF, and keyword overlap.
• Matched each job requirement to the strongest resume evidence using cosine similarity and surfaced gaps.
• Added PDF resume parsing using PyPDF2 with multi-page extraction and text cleaning pipeline.
• Built a Batch Mode to rank one resume against up to 6 job descriptions simultaneously.
• Integrated GPT-3.5 via prompt engineering to generate 5 personalized, actionable resume improvement tips.
• Visualized results using an interactive Streamlit dashboard with Plotly grouped bar and donut charts.
```

---

## 🔧 Further Extensions

- Export batch results as PDF report
- ATS keyword highlighting in resume text
- Job scraper to auto-fetch JDs from LinkedIn/Naukri

## Memory Measurement

After installing runtime dependencies, run `python scripts/measure_memory.py` to load
the model, perform one sample analysis, and print process peak RSS in MB. Model files
are downloaded from the model registry on first use.

---

*Portfolio project — Ashutosh Gupta, B.E. Computer Engineering, Thapar Institute 2026*
