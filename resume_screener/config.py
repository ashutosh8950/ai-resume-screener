"""Environment-backed application settings."""

import os


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


MAX_UPLOAD_MB = _positive_int("MAX_UPLOAD_MB", 5)
MAX_PDF_PAGES = _positive_int("MAX_PDF_PAGES", 10)
MAX_TEXT_CHARS = _positive_int("MAX_TEXT_CHARS", 20000)
MAX_BATCH_JDS = max(2, _positive_int("MAX_BATCH_JDS", 6))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    LOG_LEVEL = "INFO"

EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
SEMANTIC_WEIGHT = 0.55
SKILL_WEIGHT = 0.30
KEYWORD_WEIGHT = 0.075
TFIDF_WEIGHT = 0.075
FALLBACK_SKILL_WEIGHT = 0.50
FALLBACK_KEYWORD_WEIGHT = 0.25
FALLBACK_TFIDF_WEIGHT = 0.25
PARTIAL_MATCH_THRESHOLD = 0.5711
STRONG_MATCH_THRESHOLD = 0.7068