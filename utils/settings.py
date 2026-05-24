"""Central configuration for the Resume ATS Analyzer.

Paths, file-size limits, and shared constants live here so other modules
do not hard-code them.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "data" / "reports"
SAMPLES_DIR = DATA_DIR / "samples"
LOG_FILE = DATA_DIR / "analyzer.log"
HISTORY_CSV = DATA_DIR / "analysis_history.csv"

# Ensure runtime directories exist on import.
for directory in (DATA_DIR, REPORTS_DIR, SAMPLES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# File handling
MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB; rejects suspiciously large uploads.

# NLP knobs
# Keeping the vocabulary size bounded keeps TF-IDF fast and resistant
# to noise from rare tokens.
MAX_TFIDF_FEATURES = 5000
NGRAM_RANGE = (1, 2)  # Capture single words and short phrases like "data science".

# Composite ATS score = COMPOSITE_WEIGHT_COSINE * cosine_pct
#                     + COMPOSITE_WEIGHT_COVERAGE * keyword_coverage_pct
# The two weights must sum to 1.0.
COMPOSITE_WEIGHT_COSINE = 0.70
COMPOSITE_WEIGHT_COVERAGE = 0.30

# A small, opinionated set of skills we treat as "important" if they appear
# in the job description. The list is intentionally short — the keyword
# extractor will discover the rest from the JD itself.
COMMON_TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "django", "flask", "fastapi", "spring", "react", "vue", "angular",
    "node.js", "express",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "linux", "git", "ci/cd", "jenkins", "github actions",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "nlp", "machine learning", "deep learning", "data analysis",
    "tableau", "power bi", "excel",
    "rest", "graphql", "microservices", "agile", "scrum",
}
