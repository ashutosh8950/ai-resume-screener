"""Resume/job-description analysis."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_screener import embeddings
from resume_screener.config import (
    EMBEDDINGS_ENABLED,
    FALLBACK_KEYWORD_WEIGHT,
    FALLBACK_SKILL_WEIGHT,
    FALLBACK_TFIDF_WEIGHT,
    KEYWORD_WEIGHT,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
    TFIDF_WEIGHT,
)
from resume_screener.logger import get_logger
from resume_screener.skills import extract_skills, extract_technical_skills


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

class ResumeAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )

    def preprocess(self, text: str) -> str:
        text = embeddings.normalize_text(text).lower()
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

    def prepare_job_description(self, jd: str, precompute_embeddings: bool = False) -> dict:
        requirements, ignored_lines = embeddings.filter_job_description(jd)
        prepared = {
            "requirements": requirements,
            "ignored_lines": ignored_lines,
            "filtered_jd": "\n".join(item["text"] for item in requirements),
            "query_vectors": None,
            "embedding_message": "",
        }
        if not EMBEDDINGS_ENABLED:
            prepared["embedding_message"] = (
                "Semantic embeddings are disabled; the 50% skill coverage + "
                "25% keyword + 25% TF-IDF fallback was used."
            )
        elif requirements and precompute_embeddings:
            try:
                prepared["query_vectors"] = embeddings.embed_requirements(requirements)
            except embeddings.EmbeddingMemoryError as error:
                logger.info("Semantic embeddings skipped: %s", error)
                prepared["embedding_message"] = str(error)
            except Exception:
                logger.exception("Semantic requirement embedding failed; using fallback scores")
                prepared["embedding_message"] = (
                    "Semantic embeddings could not be loaded or run; the 50% skill coverage + "
                    "25% keyword + 25% TF-IDF fallback was used."
                )
        return prepared

    def analyze(self, resume: str, jd: str, prepared_job_description: dict | None = None) -> dict:
        prepared = prepared_job_description or self.prepare_job_description(jd)
        requirements = prepared["requirements"]
        ignored_lines = prepared["ignored_lines"]
        filtered_jd = prepared["filtered_jd"]
        tfidf = self.tfidf_similarity(resume, filtered_jd)
        keyword = self.keyword_score(resume, filtered_jd)
        semantic = 0.0
        match_result = {
            "strong_matches": [],
            "partial_matches": [],
            "gaps": [],
            "general_expectations": [],
            "ignored_lines": ignored_lines,
            "raw_match_scores": [],
        }
        embeddings_used = False
        embedding_message = prepared["embedding_message"]
        if not EMBEDDINGS_ENABLED:
            pass
        elif embedding_message:
            pass
        else:
            try:
                match_kwargs = {
                    "requirements": requirements,
                    "ignored_lines": ignored_lines,
                }
                if prepared["query_vectors"] is not None:
                    match_kwargs["prepared_query_vectors"] = prepared["query_vectors"]
                match_result = embeddings.match_requirements(resume, filtered_jd, **match_kwargs)
                semantic = match_result["semantic_score"]
                embeddings_used = bool(requirements)
                if not requirements:
                    embedding_message = "No requirement-focused JD lines remained after filtering."
            except embeddings.EmbeddingMemoryError as error:
                logger.info("Semantic embeddings skipped: %s", error)
                embedding_message = str(error)
            except Exception:
                logger.exception("Semantic embedding analysis failed; using fallback scores")
                embedding_message = (
                    "Semantic embeddings could not be loaded or run; the 50% skill coverage + "
                    "25% keyword + 25% TF-IDF fallback was used."
                )

        resume_skills = set(extract_technical_skills(resume))
        jd_skills = self.extract_skills(filtered_jd)
        required_skill_groups = []
        seen_skill_groups = set()
        for item in requirements:
            if item.get("optional", False):
                continue
            groups = item.get("skill_groups") or embeddings.technical_skill_groups(item["text"])
            for group in groups:
                normalized_group = tuple(dict.fromkeys(group))
                if normalized_group and normalized_group not in seen_skill_groups:
                    seen_skill_groups.add(normalized_group)
                    required_skill_groups.append(list(normalized_group))
        required_technical_skills = list(dict.fromkeys(
            skill for group in required_skill_groups for skill in group
        ))
        matched_technical_skills = list(dict.fromkeys(
            skill
            for group in required_skill_groups
            if set(group).intersection(resume_skills)
            for skill in group
            if skill in resume_skills
        ))
        missing_technical_skills = list(dict.fromkeys(
            skill
            for group in required_skill_groups
            if not set(group).intersection(resume_skills)
            for skill in group
        ))
        skill_coverage = (
            sum(bool(set(group).intersection(resume_skills)) for group in required_skill_groups)
            / len(required_skill_groups)
            if required_skill_groups else 0.0
        )
        if embeddings_used:
            overall = round(
                SEMANTIC_WEIGHT * semantic
                + SKILL_WEIGHT * skill_coverage
                + KEYWORD_WEIGHT * keyword
                + TFIDF_WEIGHT * tfidf,
                4,
            )
            scoring_mode = "semantic_embeddings"
        else:
            overall = round(
                FALLBACK_SKILL_WEIGHT * skill_coverage
                + FALLBACK_KEYWORD_WEIGHT * keyword
                + FALLBACK_TFIDF_WEIGHT * tfidf,
                4,
            )
            scoring_mode = "tfidf_keyword_fallback"
        return {
            "tfidf_score": tfidf,
            "keyword_score": keyword,
            "semantic_score": semantic,
            "skill_coverage": skill_coverage,
            "overall_score": overall,
            "matched_skills": matched_technical_skills,
            "missing_skills": missing_technical_skills,
            "resume_token_count": len(self.tokenize(resume)),
            "jd_token_count": len(self.tokenize(filtered_jd)),
            "raw_jd_token_count": len(self.tokenize(jd)),
            "requirement_count": len(requirements),
            "required_technical_skills": required_technical_skills,
                        "required_technical_skill_groups": required_skill_groups,
            "matched_technical_skills": matched_technical_skills,
            "missing_technical_skills": missing_technical_skills,
            "soft_skills_detected": [
                skill for skill in jd_skills if skill not in required_technical_skills
            ],
            "embeddings_used": embeddings_used,
            "embedding_message": embedding_message,
            "strong_matches": match_result["strong_matches"],
            "partial_matches": match_result["partial_matches"],
            "gaps": match_result["gaps"],
            "general_expectations": match_result.get("general_expectations", []),
            "ignored_lines": match_result["ignored_lines"],
            "raw_match_scores": match_result["raw_match_scores"],
            "scoring_mode": scoring_mode,
        }
