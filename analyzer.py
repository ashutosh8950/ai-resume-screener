"""
analyzer.py  (v2 — upgraded)
Core NLP engine:
  - TF-IDF cosine similarity       (sklearn)
  - Keyword match score            (custom tokenizer)
  - Word2Vec semantic similarity   (gensim) ← NEW
  - Skill extraction               (regex)
  - GPT improvement suggestions    (OpenAI)
"""

import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

TECH_SKILLS = [
    "python", "java", "javascript", "typescript", "sql", "r", "c++", "c", "go", "rust", "scala",
    "tensorflow", "keras", "pytorch", "scikit-learn", "sklearn", "xgboost", "lightgbm",
    "nlp", "bert", "gpt", "t5", "transformers", "spacy", "nltk", "word2vec", "glove",
    "fasttext", "tfidf", "tf-idf", "word embeddings", "sentence transformers", "langchain",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau", "power bi",
    "machine learning", "deep learning", "neural networks", "computer vision",
    "random forest", "logistic regression", "svm", "naive bayes", "gradient boosting",
    "classification", "regression", "clustering", "recommendation", "reinforcement learning",
    "git", "github", "docker", "kubernetes", "aws", "gcp", "azure", "mlflow", "airflow",
    "flask", "fastapi", "streamlit", "jupyter", "linux",
    "mysql", "postgresql", "mongodb", "redis", "sqlite",
    "speech recognition", "text to speech", "tts", "stt", "gtts", "pyttsx3",
    "api", "rest api", "opencv", "hadoop", "spark", "data visualization",
    "prompt engineering", "openai", "hugging face", "data mining", "statistics",
    "data analysis", "feature engineering", "model deployment",
]

STOPWORDS = {
    "the","and","or","in","of","to","a","an","is","are","for","with","that",
    "this","we","you","your","our","will","be","have","has","can","on","at",
    "by","from","as","it","its","also","would","should","must","use","using",
    "experience","work","working","years","ability","strong","good","knowledge",
    "including","required","preferred","responsibilities","requirements",
}


class ResumeAnalyzer:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        self._w2v_model = None

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self, text: str) -> list:
        if GENSIM_AVAILABLE:
            return simple_preprocess(text, deacc=True, min_len=2)
        return [w for w in self.preprocess(text).split() if w not in STOPWORDS]

    def tfidf_similarity(self, resume: str, jd: str) -> float:
        try:
            matrix = self.vectorizer.fit_transform(
                [self.preprocess(resume), self.preprocess(jd)]
            )
            return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        except Exception:
            return 0.0

    def keyword_score(self, resume: str, jd: str) -> float:
        resume_lower = resume.lower()
        jd_lower     = jd.lower()
        jd_tokens    = set(
            w for w in re.findall(r'\b[a-z][a-z0-9\+\#\-\.]{1,}\b', jd_lower)
            if w not in STOPWORDS
        )
        if not jd_tokens:
            return 0.0
        matched = sum(1 for t in jd_tokens if t in resume_lower)
        return matched / len(jd_tokens)

    def word2vec_similarity(self, resume: str, jd: str) -> float:
        """
        Trains Word2Vec on resume+JD corpus.
        Computes document vectors by averaging word vectors,
        then returns cosine similarity — same concept as
        word embeddings used in the Fake News Detector project.
        """
        if not GENSIM_AVAILABLE:
            return 0.0

        r_tokens = self.tokenize(resume)
        j_tokens = self.tokenize(jd)

        if not r_tokens or not j_tokens:
            return 0.0

        try:
            corpus = [r_tokens, j_tokens]
            model  = Word2Vec(
                sentences=corpus,
                vector_size=100,
                window=5,
                min_count=1,
                workers=2,
                epochs=50,
                seed=42
            )
            self._w2v_model = model

            def doc_vector(tokens):
                vecs = [model.wv[w] for w in tokens if w in model.wv]
                return np.mean(vecs, axis=0) if vecs else np.zeros(100)

            rv  = doc_vector(r_tokens).reshape(1, -1)
            jv  = doc_vector(j_tokens).reshape(1, -1)
            sim = cosine_similarity(rv, jv)[0][0]
            return float(np.clip(sim, 0, 1))
        except Exception:
            return 0.0

    def get_top_w2v_terms(self, jd: str, topn: int = 8) -> list:
        if not GENSIM_AVAILABLE or self._w2v_model is None:
            return []
        try:
            candidates = [t for t in self.tokenize(jd) if t in self._w2v_model.wv][:20]
            if not candidates:
                return []
            similar = self._w2v_model.wv.most_similar(candidates[0], topn=topn)
            return [(word, round(score, 3)) for word, score in similar]
        except Exception:
            return []

    def extract_skills(self, text: str) -> list:
        text_lower = text.lower()
        return [
            skill for skill in TECH_SKILLS
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)
        ]

    def analyze(self, resume: str, jd: str) -> dict:
        tfidf     = self.tfidf_similarity(resume, jd)
        kw        = self.keyword_score(resume, jd)
        w2v       = self.word2vec_similarity(resume, jd)
        w2v_terms = self.get_top_w2v_terms(jd)

        # TF-IDF 35% + Keyword 35% + W2V 30% when gensim available
        overall = (
            round(tfidf * 0.35 + kw * 0.35 + w2v * 0.30, 4)
            if w2v > 0
            else round(tfidf * 0.50 + kw * 0.50, 4)
        )

        resume_skills = self.extract_skills(resume)
        jd_skills     = self.extract_skills(jd)

        return {
            "tfidf_score":        tfidf,
            "keyword_score":      kw,
            "w2v_score":          w2v,
            "w2v_terms":          w2v_terms,
            "overall_score":      overall,
            "matched_skills":     [s for s in jd_skills if s in resume_skills],
            "missing_skills":     [s for s in jd_skills if s not in resume_skills],
            "resume_token_count": len(self.preprocess(resume).split()),
            "jd_token_count":     len(self.preprocess(jd).split()),
            "w2v_available":      GENSIM_AVAILABLE,
        }

    def get_gpt_suggestions(self, resume: str, jd: str, results: dict, api_key: str) -> str:
        openai.api_key = api_key
        overall_pct    = int(results["overall_score"] * 100)
        matched        = ", ".join(results["matched_skills"]) or "none detected"
        missing        = ", ".join(results["missing_skills"]) or "none"

        prompt = f"""You are an expert resume coach helping a Computer Engineering student (2026 graduate) improve their resume.

MATCH ANALYSIS:
- Overall Score: {overall_pct}%
- TF-IDF Similarity: {int(results['tfidf_score'] * 100)}%
- Keyword Match: {int(results['keyword_score'] * 100)}%
- Word2Vec Semantic Score: {int(results['w2v_score'] * 100)}%
- Matched Skills: {matched}
- Missing Skills: {missing}

RESUME (truncated): {resume[:1000]}
JOB DESCRIPTION (truncated): {jd[:800]}

Give exactly 5 specific, actionable improvement suggestions numbered 1-5. Each 2-3 sentences. Be direct and practical."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional resume coach. Be specific, direct, practical."},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except openai.error.AuthenticationError:
            return "❌ Invalid API key. Please check your OpenAI API key in the sidebar."
        except openai.error.RateLimitError:
            return "❌ Rate limit hit. Please wait a moment and try again."
        except Exception as e:
            return f"❌ Error: {str(e)}"
