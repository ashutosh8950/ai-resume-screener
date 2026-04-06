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
