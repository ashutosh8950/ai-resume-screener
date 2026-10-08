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