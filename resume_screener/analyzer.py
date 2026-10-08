"""Resume/job-description analysis and optional OpenAI suggestions."""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_screener.logger import get_logger
from resume_screener.skills import extract_skills


logger = get_logger("analyzer")
STOPWORDS = {
    "the", "and", "or", "in", "of", "to", "a", "an", "is", "are", "for", "with", "that",
    "this", "we", "you", "your", "our", "will", "be", "have", "has", "can", "on", "at",
    "by", "from", "as", "it", "its", "also", "would", "should", "must", "use", "using",
    "experience", "work", "working", "years", "ability", "strong", "good", "knowledge",
    "including", "required", "preferred", "responsibilities", "requirements",
}
TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:\.[A-Za-z0-9]+|[A-Za-z0-9]+(?:[+#-][A-Za-z0-9+#-]*|\.[A-Za-z0-9]+)*)(?![A-Za-z0-9_])"
)

try:
    from gensim.models import Word2Vec

    GENSIM_AVAILABLE = True
    GENSIM_IMPORT_ERROR = ""
except Exception as error:
    Word2Vec = None
    GENSIM_AVAILABLE = False
    GENSIM_IMPORT_ERROR = str(error)
    logger.exception("Word2Vec is unavailable")


class OpenAIClient:
    """Small per-request REST client compatible with the pinned OpenAI SDK version."""

    endpoint = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_chat_completion(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        body = json.dumps({
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


class ResumeAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        self._w2v_model = None
        self._w2v_status = "not_run"

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"[^\w\s\+\#\.]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text) if token.lower() not in STOPWORDS]

    def tfidf_similarity(self, resume: str, jd: str) -> float:
        try:
            matrix = self.vectorizer.fit_transform([self.preprocess(resume), self.preprocess(jd)])
            return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        except Exception:
            logger.exception("TF-IDF similarity calculation failed")
            return 0.0

    def keyword_score(self, resume: str, jd: str) -> float:
        resume_tokens = set(self.tokenize(resume))
        jd_tokens = set(self.tokenize(jd))
        if not jd_tokens:
            return 0.0
        return len(resume_tokens & jd_tokens) / len(jd_tokens)

    def word2vec_similarity(self, resume: str, jd: str) -> float:
        self._w2v_model = None
        if not GENSIM_AVAILABLE:
            self._w2v_status = "unavailable"
            return 0.0

        resume_tokens = self.tokenize(resume)
        jd_tokens = self.tokenize(jd)
        if not resume_tokens or not jd_tokens:
            self._w2v_status = "skipped_empty"
            return 0.0

        try:
            model = Word2Vec(
                sentences=[resume_tokens, jd_tokens],
                vector_size=100,
                window=5,
                min_count=1,
                workers=2,
                epochs=50,
                seed=42,
            )
            self._w2v_model = model
            resume_vector = np.mean([model.wv[token] for token in resume_tokens], axis=0).reshape(1, -1)
            jd_vector = np.mean([model.wv[token] for token in jd_tokens], axis=0).reshape(1, -1)
            score = float(cosine_similarity(resume_vector, jd_vector)[0][0])
            self._w2v_status = "ok"
            return float(np.clip(score, 0, 1))
        except Exception:
            self._w2v_status = "error"
            logger.exception("Word2Vec similarity calculation failed")
            return 0.0

    def get_top_w2v_terms(self, jd: str, topn: int = 8) -> list[tuple[str, float]]:
        if self._w2v_model is None:
            return []
        try:
            candidates = [token for token in self.tokenize(jd) if token in self._w2v_model.wv][:20]
            if not candidates:
                return []
            similar = self._w2v_model.wv.most_similar(candidates[0], topn=topn)
            return [(word, round(score, 3)) for word, score in similar]
        except Exception:
            logger.exception("Unable to extract related Word2Vec terms")
            return []

    def extract_skills(self, text: str) -> list[str]:
        return extract_skills(text)

    def analyze(self, resume: str, jd: str) -> dict:
        tfidf = self.tfidf_similarity(resume, jd)
        keyword = self.keyword_score(resume, jd)
        w2v = self.word2vec_similarity(resume, jd)
        w2v_terms = self.get_top_w2v_terms(jd)
        w2v_used = w2v > 0

        overall = (
            round(tfidf * 0.35 + keyword * 0.35 + w2v * 0.30, 4)
            if w2v_used
            else round(tfidf * 0.50 + keyword * 0.50, 4)
        )
        if self._w2v_status == "unavailable":
            w2v_message = "Word2Vec is unavailable; the existing TF-IDF/keyword fallback weights were used."
        elif self._w2v_status == "error":
            w2v_message = "Word2Vec could not run; the existing TF-IDF/keyword fallback weights were used."
        elif self._w2v_status == "skipped_empty":
            w2v_message = "Word2Vec was skipped because an input had no tokens; fallback weights were used."
        elif not w2v_used:
            w2v_message = "Word2Vec returned zero similarity; the existing fallback weights were used."
        else:
            w2v_message = "Word2Vec was included in the overall score."

        resume_skills = self.extract_skills(resume)
        jd_skills = self.extract_skills(jd)
        return {
            "tfidf_score": tfidf,
            "keyword_score": keyword,
            "w2v_score": w2v,
            "w2v_terms": w2v_terms,
            "overall_score": overall,
            "matched_skills": [skill for skill in jd_skills if skill in resume_skills],
            "missing_skills": [skill for skill in jd_skills if skill not in resume_skills],
            "resume_token_count": len(self.tokenize(resume)),
            "jd_token_count": len(self.tokenize(jd)),
            "w2v_available": self._w2v_status == "ok",
            "w2v_status": self._w2v_status,
            "w2v_used": w2v_used,
            "w2v_message": w2v_message,
            "scoring_method": "weighted_with_word2vec" if w2v_used else "tfidf_keyword_fallback",
        }

    def get_gpt_suggestions(self, resume: str, jd: str, results: dict, api_key: str) -> str:
        overall_pct = int(results["overall_score"] * 100)
        matched = ", ".join(results["matched_skills"]) or "none detected"
        missing = ", ".join(results["missing_skills"]) or "none"
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
        client = OpenAIClient(api_key)
        try:
            response = client.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a professional resume coach. Be specific, direct, practical."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.7,
            )
            return response["choices"][0]["message"]["content"].strip()
        except HTTPError as error:
            logger.warning("OpenAI request returned HTTP %s", error.code)
            if error.code == 401:
                return "Invalid API key. Please check the key and try again."
            if error.code == 429:
                return "OpenAI rate limit reached. Please wait and try again."
            return "OpenAI could not generate suggestions right now. Please try again later."
        except (URLError, TimeoutError, KeyError, ValueError):
            logger.exception("OpenAI suggestions request failed")
            return "OpenAI could not generate suggestions right now. Please try again later."
        except Exception:
            logger.exception("Unexpected error while generating OpenAI suggestions")
            return "OpenAI could not generate suggestions right now. Please try again later."