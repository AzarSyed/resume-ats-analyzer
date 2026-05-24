"""Run an analysis from the command line.

Examples::

    python main.py --resume data/samples/sample_resume.txt --jd data/samples/job_description.txt
    python main.py --resume my_resume.pdf --jd job.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analyzer.ats_score import compose_score, compute_match
from analyzer.keyword_analysis import analyse_keywords
from parser.resume_parser import extract_text
from reports.report_generator import save_report
from utils.logger import get_logger

log = get_logger("main")


def _read_resume(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text(path)
    return path.read_text(encoding="utf-8")


def run(resume_path: Path, jd_path: Path) -> None:
    log.info("Reading resume: %s", resume_path)
    resume_text = _read_resume(resume_path)

    log.info("Reading job description: %s", jd_path)
    jd_text = jd_path.read_text(encoding="utf-8")

    match = compute_match(resume_text, jd_text)
    analysis = analyse_keywords(resume_text, jd_text)
    composite = compose_score(match.percent, analysis.coverage_pct)

    log.info("=" * 60)
    log.info("FINAL ATS SCORE:    %.2f%%   (%s)", composite.composite_pct, composite.verdict)
    log.info("  Cosine similarity: %.2f%%  (weight 70%%)", composite.cosine_pct)
    log.info("  Keyword coverage:  %.2f%%  (weight 30%%)", composite.coverage_pct)
    log.info("=" * 60)
    log.info("Matched (%d): %s", len(analysis.matched), ", ".join(analysis.matched[:10]))
    log.info("Missing tech: %s", ", ".join(analysis.missing_tech) or "(none)")
    for tip in analysis.suggestions:
        log.info("Tip: %s", tip)

    saved = save_report(match, analysis, resume_path.name, jd_text, composite)
    log.info("Report saved: %s", saved["report_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume ATS Analyzer (CLI)")
    parser.add_argument("--resume", required=True, type=Path, help="Path to resume (PDF or TXT)")
    parser.add_argument("--jd", required=True, type=Path, help="Path to job description (TXT)")
    args = parser.parse_args()
    run(args.resume, args.jd)


if __name__ == "__main__":
    main()
