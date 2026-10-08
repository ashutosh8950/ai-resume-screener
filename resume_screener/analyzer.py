"""Resume/job-description analysis and optional OpenAI suggestions."""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_screener import embeddings
from resume_screener.config import (
    EMBEDDINGS_ENABLED,
    FALLBACK_KEYWORD_WEIGHT,
    FALLBACK_TFIDF_WEIGHT,
    KEYWORD_WEIGHT,
    SEMANTIC_WEIGHT,
    TFIDF_WEIGHT,
)
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

    def preprocess(self, text: str) -> str:
        text = embeddings.clean_text(text).lower()
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

    def extract_skills(self, text: str) -> list[str]:
        return extract_skills(text)

    def analyze(self, resume: str, jd: str) -> dict:
        requirements, ignored_lines = embeddings.filter_job_description(jd)
        filtered_jd = "\n".join(requirements)
        tfidf = self.tfidf_similarity(resume, filtered_jd)
        keyword = self.keyword_score(resume, filtered_jd)
        semantic = 0.0
        match_result = {
            "strong_matches": [],
            "partial_matches": [],
            "gaps": [],
            "ignored_lines": ignored_lines,
            "raw_match_scores": [],
        }
        embeddings_used = False
        embedding_message = ""
        if not EMBEDDINGS_ENABLED:
            embedding_message = "Semantic embeddings are disabled; the 50% keyword + 50% TF-IDF fallback was used."
        else:
            try:
                match_result = embeddings.match_requirements(
                    resume,
                    filtered_jd,
                    requirements=requirements,
                    ignored_lines=ignored_lines,
                )
                semantic = match_result["semantic_score"]
                embeddings_used = bool(requirements)
                if not requirements:
                    embedding_message = "No requirement-focused JD lines remained after filtering."
            except Exception:
                logger.exception("Semantic embedding analysis failed; using fallback scores")
                embedding_message = (
                    "Semantic embeddings could not be loaded or run; the 50% keyword + "
                    "50% TF-IDF fallback was used."
                )

        if embeddings_used:
            overall = round(
                SEMANTIC_WEIGHT * semantic
                + KEYWORD_WEIGHT * keyword
                + TFIDF_WEIGHT * tfidf,
                4,
            )
            scoring_mode = "semantic_embeddings"
        else:
            overall = round(
                FALLBACK_KEYWORD_WEIGHT * keyword + FALLBACK_TFIDF_WEIGHT * tfidf,
                4,
            )
            scoring_mode = "tfidf_keyword_fallback"

        resume_skills = self.extract_skills(resume)
        jd_skills = self.extract_skills(filtered_jd)
        return {
            "tfidf_score": tfidf,
            "keyword_score": keyword,
            "semantic_score": semantic,
            "overall_score": overall,
            "matched_skills": [skill for skill in jd_skills if skill in resume_skills],
            "missing_skills": [skill for skill in jd_skills if skill not in resume_skills],
            "resume_token_count": len(self.tokenize(resume)),
            "jd_token_count": len(self.tokenize(filtered_jd)),
            "raw_jd_token_count": len(self.tokenize(jd)),
            "requirement_count": len(requirements),
            "embeddings_used": embeddings_used,
            "embedding_message": embedding_message,
            "strong_matches": match_result["strong_matches"],
            "partial_matches": match_result["partial_matches"],
            "gaps": match_result["gaps"],
            "ignored_lines": match_result["ignored_lines"],
            "raw_match_scores": match_result["raw_match_scores"],
            "scoring_mode": scoring_mode,
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
- Semantic Match Score: {int(results['semantic_score'] * 100)}%
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