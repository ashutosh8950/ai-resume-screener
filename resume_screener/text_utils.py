"""Small text-normalization helpers used at UI input boundaries."""

import re


_BULLET_REPLACEMENTS = {
    ord(character): "- "
    for character in (
        "\uf0b7", "\uf0a7", "\u2022", "\u25aa", "\u25ab", "\u25cf",
        "\u2023", "\u2043", "\u2219", "\u25e6", "\u2023", "\u27a2",
    )
}
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")


def clean_jd_text(text: str) -> str:
    """Normalize copied or extracted job-description text without changing words."""
    normalized = text.translate(_BULLET_REPLACEMENTS)
    normalized = _CONTROL_RE.sub("", normalized)
    return _PRIVATE_USE_RE.sub("", normalized)
