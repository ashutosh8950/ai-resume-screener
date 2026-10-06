# 📄 AI Resume Screener v2

> Compare a resume against a job description and get a weighted match score, skill-gap breakdown, and five targeted improvement suggestions.

[https://ai-resume-screener-w6h3.onrender.com](https://ai-resume-screener-w6h3.onrender.com) · [https://ashutosh8950.github.io/Portfolio/](https://ashutosh8950.github.io/Portfolio/)

*Free-tier hosting: the first load can take about a minute while the service wakes up.*

Match scoring runs without any key. GPT-3.5 suggestions need your own OpenAI API key, entered in the sidebar for that session.

An advanced resume-job description matcher using **TF-IDF**, **Word2Vec**, **keyword similarity**, and **GPT-3.5** suggestions.

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
├── analyzer.py       ← TF-IDF, keyword match, Word2Vec, GPT
├── pdf_parser.py     ← PDF text extraction (PyPDF2)     ← NEW
├── visualizer.py     ← Plotly charts (updated for v2)
├── requirements.txt  ← All dependencies
└── README.md
```

---

## 🛠️ Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open browser
# http://localhost:8501
```

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

### 3. Word2Vec Semantic Similarity (30%) ← NEW
- Trains `Word2Vec` on resume+JD as a small corpus
- Builds 100-dim vectors for each word
- Averages word vectors → document vector
- Cosine similarity between document vectors
- **Advantage**: catches synonyms and semantic matches TF-IDF misses
  - Example: "deep learning" ≈ "neural networks" ≈ "pytorch"

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

## 🔧 Further Extensions

- Export batch results as PDF report
- BERT/sentence-transformers for better semantic similarity
- ATS keyword highlighting in resume text
- Job scraper to auto-fetch JDs from LinkedIn/Naukri

---

*Portfolio project — Ashutosh Gupta, B.E. Computer Engineering, Thapar Institute 2026*
