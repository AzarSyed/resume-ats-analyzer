"""ATS match score using TF-IDF and cosine similarity.

The score is the cosine similarity of the TF-IDF vector for the resume
and the TF-IDF vector for the job description, expressed as a percent.
Both texts are normalised by ``text_cleaner.clean_text`` first so the
comparison is on lemmatised, stopword-free tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analyzer.text_cleaner import clean_text
from utils.logger import get_logger
from utils.settings import (
    COMPOSITE_WEIGHT_COSINE,
    COMPOSITE_WEIGHT_COVERAGE,
    MAX_TFIDF_FEATURES,
    NGRAM_RANGE,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a resume vs. job-description comparison."""

    score: float  # 0.0 - 1.0
    percent: float  # 0.0 - 100.0
    resume_tokens: int
    jd_tokens: int


@dataclass(frozen=True)
class CompositeScore:
    """Final ATS score blending cosine similarity and keyword coverage."""

    cosine_pct: float
    coverage_pct: float
    composite_pct: float
    verdict: str


def compose_score(cosine_pct: float, coverage_pct: float) -> CompositeScore:
    """Combine cosine and coverage into a single, more interpretable score.

    The 70/30 weighting keeps the rigorous TF-IDF signal as the dominant
    factor while ensuring resumes that hit most JD keywords still score
    well even when their overall vocabulary differs.
    """
    composite = (
        COMPOSITE_WEIGHT_COSINE * cosine_pct
        + COMPOSITE_WEIGHT_COVERAGE * coverage_pct
    )
    composite = round(max(0.0, min(100.0, composite)), 2)

    if composite >= 70:
        verdict = "Strong match"
    elif composite >= 45:
        verdict = "Reasonable match — room to improve"
    else:
        verdict = "Weak match — significant gaps"

    return CompositeScore(
        cosine_pct=round(cosine_pct, 2),
        coverage_pct=round(coverage_pct, 2),
        composite_pct=composite,
        verdict=verdict,
    )


def compute_match(resume_text: str, jd_text: str) -> MatchResult:
    """Return the cosine similarity between resume and job description.

    Raises ``ValueError`` when either input is empty after cleaning, so
    callers can show a sensible error in the UI.
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    if not cleaned_resume:
        raise ValueError("Resume text is empty after cleaning.")
    if not cleaned_jd:
        raise ValueError("Job description text is empty after cleaning.")

    # ``fit_transform`` learns the vocabulary from both documents
    # combined, then projects each into a TF-IDF vector.
    vectorizer = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=NGRAM_RANGE,
    )
    matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

    similarity = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    similarity = max(0.0, min(1.0, similarity))  # clamp to [0, 1]

    result = MatchResult(
        score=similarity,
        percent=round(similarity * 100, 2),
        resume_tokens=len(cleaned_resume.split()),
        jd_tokens=len(cleaned_jd.split()),
    )
    log.info(
        "ATS match=%.2f%% (resume=%d tokens, jd=%d tokens)",
        result.percent,
        result.resume_tokens,
        result.jd_tokens,
    )
    return result
