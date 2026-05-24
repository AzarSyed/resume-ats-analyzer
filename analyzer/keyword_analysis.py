"""Keyword and skill-gap analysis.

Given a resume and a job description, we identify:

* the keywords the JD emphasises (top TF-IDF terms),
* which of those appear in the resume (matched),
* which do not (missing — the user's improvement target),
* a few human-readable suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from analyzer.text_cleaner import clean_text, tokens
from utils.logger import get_logger
from utils.settings import COMMON_TECH_SKILLS, MAX_TFIDF_FEATURES

log = get_logger(__name__)


@dataclass(frozen=True)
class KeywordAnalysis:
    matched: list[str]
    missing: list[str]
    missing_tech: list[str]  # Curated tech skills that are missing.
    suggestions: list[str]

    @property
    def coverage(self) -> float:
        """Fraction of JD keywords present in the resume (0.0 - 1.0)."""
        total = len(self.matched) + len(self.missing)
        return (len(self.matched) / total) if total else 0.0

    @property
    def coverage_pct(self) -> float:
        return round(self.coverage * 100, 2)


def _jd_keywords(jd_text: str, top_n: int = 25) -> list[str]:
    """Return the top ``top_n`` keywords from the job description by TF-IDF."""
    cleaned = clean_text(jd_text)
    if not cleaned:
        return []

    # Fitting on a single document means IDF equals 1 for every term,
    # so TF-IDF reduces to raw term frequency. That is still a useful
    # ranking signal: terms repeated in the JD matter more.
    vectorizer = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform([cleaned])
    scores = matrix.toarray()[0]
    vocab = vectorizer.get_feature_names_out()

    # Sort terms by score descending and take the top N.
    ranked = sorted(zip(vocab, scores), key=lambda pair: pair[1], reverse=True)
    return [term for term, score in ranked[:top_n] if score > 0]


def analyse_keywords(resume_text: str, jd_text: str) -> KeywordAnalysis:
    """Compare resume against JD; return matched/missing keywords + tips."""
    jd_terms = _jd_keywords(jd_text)
    resume_tokens = set(tokens(resume_text))

    # Always also include the curated tech skills that the JD mentions —
    # these are high-signal even if their TF-IDF is modest.
    cleaned_jd_words = set(tokens(jd_text))
    curated_jd_skills = sorted(cleaned_jd_words & COMMON_TECH_SKILLS)

    candidate_terms: list[str] = []
    seen: set[str] = set()
    for term in jd_terms + curated_jd_skills:
        if term in seen:
            continue
        seen.add(term)
        candidate_terms.append(term)

    matched: list[str] = []
    missing: list[str] = []
    for term in candidate_terms:
        # Multi-word terms (bigrams) need substring matching on cleaned tokens.
        words = term.split()
        present = (
            all(word in resume_tokens for word in words)
            if len(words) > 1
            else term in resume_tokens
        )
        (matched if present else missing).append(term)

    # Promote curated tech-skill gaps so the UI can highlight them.
    missing_tech = [term for term in missing if term in COMMON_TECH_SKILLS]
    # Sort so curated skills surface first, then bigrams, then the rest.
    missing.sort(key=lambda t: (t not in COMMON_TECH_SKILLS, " " not in t, t))
    matched.sort(key=lambda t: (t not in COMMON_TECH_SKILLS, " " not in t, t))

    suggestions = _make_suggestions(matched, missing, missing_tech)
    log.info(
        "Keyword analysis: %d matched, %d missing (%d tech).",
        len(matched),
        len(missing),
        len(missing_tech),
    )
    return KeywordAnalysis(
        matched=matched,
        missing=missing,
        missing_tech=missing_tech,
        suggestions=suggestions,
    )


def _make_suggestions(
    matched: list[str],
    missing: list[str],
    missing_tech: list[str],
) -> list[str]:
    """Build a small list of plain-English improvement tips."""
    tips: list[str] = []
    if missing_tech:
        tips.append(
            "Strengthen technical coverage on: "
            + ", ".join(missing_tech[:5])
            + "."
        )
    if missing:
        non_tech_missing = [t for t in missing if t not in COMMON_TECH_SKILLS][:5]
        if non_tech_missing:
            tips.append(
                "Mirror the job description's vocabulary by referencing: "
                + ", ".join(non_tech_missing)
                + "."
            )
    if matched and len(matched) < 5:
        tips.append(
            "Expand bullet points around your strongest matches "
            f"({', '.join(matched[:3])}) with measurable outcomes."
        )
    if not tips:
        tips.append("Strong alignment. Tighten wording and quantify achievements.")
    return tips
