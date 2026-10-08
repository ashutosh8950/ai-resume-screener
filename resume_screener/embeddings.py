"""Lazy CPU embedding model and sentence-level requirement matching."""

import re
import unicodedata

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from resume_screener.config import EMBEDDING_MODEL


_embedding_model = None
_BULLET_CHARS = "•▪◦▯□■●‣⁃◆◇"
_BULLET_PATTERN = re.compile(f"[{re.escape(_BULLET_CHARS)}]")
_LINE_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
MATCH_THRESHOLD = 0.5


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\ufffd", " ")
    text = _BULLET_PATTERN.sub(" ", text)
    return _LINE_BULLET_PATTERN.sub("", text)


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text.replace("\r\n", "\n").replace("\r", "\n"))
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def get_embedding_model():
    """Create the fastembed model once, on the first requested analysis."""
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding

        _embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL,
            providers=["CPUExecutionProvider"],
        )
    return _embedding_model


def match_requirements(resume: str, job_description: str) -> tuple[float, list[dict], list[dict]]:
    requirements = split_sentences(job_description)
    evidence = split_sentences(resume)
    if not requirements:
        return 0.0, [], []
    if not evidence:
        gaps = [{"requirement": item, "score": 0.0} for item in requirements]
        return 0.0, [], gaps

    model = get_embedding_model()
    vectors = np.asarray(list(model.embed(requirements + evidence)), dtype=np.float32)
    requirement_vectors = vectors[:len(requirements)]
    evidence_vectors = vectors[len(requirements):]
    similarities = cosine_similarity(requirement_vectors, evidence_vectors)

    pairs = []
    best_scores = []
    for index, requirement in enumerate(requirements):
        best_index = int(np.argmax(similarities[index]))
        score = float(np.clip(similarities[index, best_index], 0.0, 1.0))
        best_scores.append(score)
        pairs.append({
            "requirement": requirement,
            "evidence": evidence[best_index],
            "score": round(score, 4),
        })

    gaps = sorted(
        (pair for pair in pairs if pair["score"] < MATCH_THRESHOLD),
        key=lambda pair: pair["score"],
    )
    matched = sorted(
        (pair for pair in pairs if pair["score"] >= MATCH_THRESHOLD),
        key=lambda pair: pair["score"],
        reverse=True,
    )
    semantic_score = float(np.clip(np.mean(best_scores), 0.0, 1.0))
    return semantic_score, matched, gaps