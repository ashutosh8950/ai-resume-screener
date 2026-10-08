# 📄 AI Resume Screener v2

An advanced resume-job description matcher using **TF-IDF**, **Word2Vec**, **keyword similarity**, and **GPT-3.5** suggestions — built entirely with your existing Python/NLP/ML skills.

---

## 🚀 What's New in v2

| Feature | Tech |
|---|---|
| **PDF Resume Upload** | `PyPDF2` — extract text from any resume PDF |
| **Word2Vec Semantic Similarity** | `gensim` Word2Vec — meaning-level matching |
| **Batch Mode** | Compare 1 resume against up to 6 jobs at once |
| **4-score dashboard** | TF-IDF + Keywords + Word2Vec + Overall |
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

---

## 🧠 How Scoring Works

### 1. TF-IDF Similarity (35%)
- `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)`
- Cosine similarity between resume and JD vectors
- Captures term importance, not just frequency

### 2. Keyword Match (35%)
- Tokenizes JD, removes stopwords
- Checks fraction of JD keywords present in resume
- Simulates how ATS systems scan resumes

### 3. Word2Vec Similarity (30%)
- Trains `Word2Vec` on the current resume+JD pair; this is a lightweight heuristic, not a pretrained semantic model
- Builds 100-dim vectors for each word
- Averages word vectors → document vector
- Cosine similarity between document vectors
- **Advantage**: catches synonyms and semantic matches TF-IDF misses
  - Treat this score as experimental; it is not validated synonym detection

### Overall Score
```python
# With gensim:
overall = tfidf * 0.35 + keyword * 0.35 + w2v * 0.30

# Without gensim (fallback):
overall = tfidf * 0.50 + keyword * 0.50
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
- Grouped bar chart (TF-IDF / Keywords / Word2Vec / Overall per job)
- Deep-dive skill gap for the top match

---

## 📌 Resume Bullet Points (copy-paste ready)

```
AI Resume Screener v2 | Python, Scikit-learn, Gensim, PyPDF2, OpenAI, Streamlit, Plotly

• Built a resume-job matcher scoring TF-IDF (sklearn), Word2Vec semantic similarity (gensim), and keyword overlap.
• Trained Word2Vec on resume+JD corpus to capture semantic relationships beyond exact keyword matching.
• Added PDF resume parsing using PyPDF2 with multi-page extraction and text cleaning pipeline.
• Built a Batch Mode to rank one resume against up to 6 job descriptions simultaneously.
• Integrated GPT-3.5 via prompt engineering to generate 5 personalized, actionable resume improvement tips.
• Visualized results using an interactive Streamlit dashboard with Plotly grouped bar and donut charts.
```

---

## 🔧 Further Extensions

- Export batch results as PDF report
- BERT/sentence-transformers for better semantic similarity
- ATS keyword highlighting in resume text
- Job scraper to auto-fetch JDs from LinkedIn/Naukri

---

*Portfolio project — Ashutosh Gupta, B.E. Computer Engineering, Thapar Institute 2026*
