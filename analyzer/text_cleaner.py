"""Text normalisation for resumes and job descriptions.

The pipeline is intentionally simple and tuned for ATS-style comparison:
lowercase, strip noise, tokenise, drop stopwords, lemmatise. Keeping it
deterministic means the same input always produces the same tokens, so
two analysers running on different machines agree.
"""

from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils.logger import get_logger

log = get_logger(__name__)

_NLTK_PACKAGES = [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def _ensure_nltk_assets() -> None:
    """Download NLTK data once on first use.

    Streamlit Cloud's container is ephemeral, so we cannot assume the
    files exist. ``nltk.data.find`` raises ``LookupError`` when missing.
    """
    for resource_path, package in _NLTK_PACKAGES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            log.info("Downloading NLTK package: %s", package)
            nltk.download(package, quiet=True)


# Strip URLs, emails, and phone numbers before tokenising — they boost
# similarity without contributing skill signal.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{7,}\d")
# Keep word characters, spaces, hyphens, dots inside tech terms (e.g.
# "node.js", "c++"), and plus signs. Drop everything else.
_NOISE_RE = re.compile(r"[^a-z0-9+.#\-\s]")

_lemmatizer: WordNetLemmatizer | None = None
_stopwords: set[str] | None = None


def _lazy_init() -> tuple[WordNetLemmatizer, set[str]]:
    global _lemmatizer, _stopwords
    if _lemmatizer is None or _stopwords is None:
        _ensure_nltk_assets()
        _lemmatizer = WordNetLemmatizer()
        _stopwords = set(stopwords.words("english"))
    return _lemmatizer, _stopwords


def clean_text(raw: str) -> str:
    """Return a normalised, lemmatised version of ``raw``.

    Empty inputs are returned as empty strings. The function is safe to
    call on any user-provided text.
    """
    if not raw or not raw.strip():
        return ""

    lemmatizer, stop_set = _lazy_init()

    text = raw.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _NOISE_RE.sub(" ", text)

    tokens = word_tokenize(text)
    cleaned: list[str] = []
    for token in tokens:
        if token in stop_set:
            continue
        # Drop tokens that are pure punctuation or single chars.
        if len(token) < 2 and token not in {"c", "r"}:
            continue
        cleaned.append(lemmatizer.lemmatize(token))
    return " ".join(cleaned)


def tokens(raw: str) -> list[str]:
    """Return cleaned tokens as a list (used by keyword analysis)."""
    cleaned = clean_text(raw)
    return cleaned.split() if cleaned else []
