"""Produce CSV reports and append to the analysis history.

Two outputs per run:

* ``data/reports/analysis_<timestamp>.csv`` — full per-run report.
* ``data/analysis_history.csv`` — one-row summary appended for the dashboard.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from io import StringIO

import pandas as pd

from analyzer.ats_score import CompositeScore, MatchResult
from analyzer.keyword_analysis import KeywordAnalysis
from utils.helpers import now_iso
from utils.logger import get_logger
from utils.settings import HISTORY_CSV, REPORTS_DIR

log = get_logger(__name__)


def build_report_frame(
    match: MatchResult,
    analysis: KeywordAnalysis,
    resume_name: str,
    jd_preview: str,
    composite: CompositeScore | None = None,
) -> pd.DataFrame:
    """Return a tidy DataFrame describing the analysis."""
    rows: list[dict] = []
    rows.append({"section": "summary", "key": "resume", "value": resume_name})
    rows.append({"section": "summary", "key": "timestamp", "value": now_iso()})
    if composite is not None:
        rows.append({"section": "summary", "key": "ats_composite_pct", "value": composite.composite_pct})
        rows.append({"section": "summary", "key": "verdict", "value": composite.verdict})
    rows.append({"section": "summary", "key": "cosine_pct", "value": match.percent})
    rows.append({"section": "summary", "key": "coverage_pct", "value": analysis.coverage_pct})
    rows.append({"section": "summary", "key": "resume_tokens", "value": match.resume_tokens})
    rows.append({"section": "summary", "key": "jd_tokens", "value": match.jd_tokens})
    rows.append({"section": "summary", "key": "jd_preview", "value": jd_preview[:200]})

    for term in analysis.matched:
        rows.append({"section": "matched", "key": term, "value": "yes"})
    for term in analysis.missing:
        rows.append({"section": "missing", "key": term, "value": "no"})
    for term in analysis.missing_tech:
        rows.append({"section": "missing_tech", "key": term, "value": "tech"})
    for tip in analysis.suggestions:
        rows.append({"section": "suggestion", "key": "tip", "value": tip})

    return pd.DataFrame(rows, columns=["section", "key", "value"])


def report_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Return the CSV-encoded bytes of a report frame (for download buttons)."""
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def save_report(
    match: MatchResult,
    analysis: KeywordAnalysis,
    resume_name: str,
    jd_preview: str,
    composite: CompositeScore | None = None,
) -> dict:
    """Write a per-run CSV and append a history row. Returns paths/summary."""
    frame = build_report_frame(match, analysis, resume_name, jd_preview, composite)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"analysis_{stamp}.csv"
    frame.to_csv(report_path, index=False)
    log.info("Saved report to %s", report_path)

    history_row = {
        "timestamp": now_iso(),
        "resume": resume_name,
        "ats_composite_pct": composite.composite_pct if composite else None,
        "cosine_pct": match.percent,
        "coverage_pct": analysis.coverage_pct,
        "resume_tokens": match.resume_tokens,
        "jd_tokens": match.jd_tokens,
        "matched_count": len(analysis.matched),
        "missing_count": len(analysis.missing),
        "missing_tech": ";".join(analysis.missing_tech),
        "report_file": report_path.name,
    }
    # Read-then-write keeps the file schema-consistent. If the existing
    # CSV has fewer columns, pandas back-fills the missing ones with NaN
    # automatically; the file is then rewritten with the unified header.
    existing = load_history()
    combined = pd.concat([existing, pd.DataFrame([history_row])], ignore_index=True)
    combined.to_csv(HISTORY_CSV, index=False)
    log.info("Wrote %d row(s) to %s", len(combined), HISTORY_CSV)

    return {"report_path": str(report_path), "history": history_row}


_HISTORY_COLUMNS = [
    "timestamp", "resume",
    "ats_composite_pct", "cosine_pct", "coverage_pct",
    "resume_tokens", "jd_tokens",
    "matched_count", "missing_count",
    "missing_tech", "report_file",
]


def load_history() -> pd.DataFrame:
    """Load all historical analyses (used by the dashboard).

    Tolerates schema drift across older runs: if the file has fewer or
    differently named columns, missing ones are filled with NaN. Lines
    that cannot be parsed at all are skipped with a warning so a single
    bad row never breaks the dashboard.
    """
    if not HISTORY_CSV.exists():
        return pd.DataFrame(columns=_HISTORY_COLUMNS)

    try:
        frame = pd.read_csv(HISTORY_CSV)
    except pd.errors.ParserError as exc:
        log.warning("Could not parse %s cleanly (%s); skipping bad lines.", HISTORY_CSV, exc)
        frame = pd.read_csv(HISTORY_CSV, on_bad_lines="skip", engine="python")

    # Ensure all expected columns are present.
    for column in _HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    # Reorder so downstream code sees a stable schema.
    frame = frame[_HISTORY_COLUMNS]

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame
