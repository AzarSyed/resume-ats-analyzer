"""Streamlit UI for the Resume ATS Analyzer.

Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow ``streamlit run dashboard/app.py`` to find sibling packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyzer.ats_score import compose_score, compute_match  # noqa: E402
from analyzer.keyword_analysis import analyse_keywords  # noqa: E402
from parser.resume_parser import ResumeParseError, extract_text  # noqa: E402
from reports.report_generator import (  # noqa: E402
    build_report_frame,
    load_history,
    report_csv_bytes,
    save_report,
)
from utils.settings import SAMPLES_DIR  # noqa: E402


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon=":memo:",
    layout="wide",
)


def _load_sample_jd() -> str:
    """Return a packaged sample job description if present, else a default."""
    path = SAMPLES_DIR / "job_description.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "We are hiring a Senior Python Engineer. Required skills: Python, "
        "Django, REST APIs, PostgreSQL, Docker, Kubernetes, AWS, CI/CD, "
        "unit testing, and strong communication."
    )


def _resume_text_from_upload(uploaded_file) -> tuple[str, str]:
    """Return (resume_text, resume_name) from a Streamlit upload."""
    name = uploaded_file.name
    data = uploaded_file.getvalue()
    text = extract_text(data)
    return text, name


def _resume_text_from_sample() -> tuple[str, str]:
    """Use the packaged sample resume when the user has not uploaded one."""
    sample_pdf = SAMPLES_DIR / "sample_resume.pdf"
    sample_txt = SAMPLES_DIR / "sample_resume.txt"
    if sample_pdf.exists():
        return extract_text(sample_pdf), sample_pdf.name
    if sample_txt.exists():
        return sample_txt.read_text(encoding="utf-8"), sample_txt.name
    raise FileNotFoundError("No sample resume packaged with the project.")


def _render_inputs() -> tuple[str, str, str]:
    """Render upload + JD inputs. Returns (resume_text, resume_name, jd_text)."""
    st.subheader("1. Inputs")
    use_sample = st.checkbox("Use packaged sample resume", value=True)

    resume_text = ""
    resume_name = ""

    if use_sample:
        try:
            resume_text, resume_name = _resume_text_from_sample()
            st.info(f"Using sample resume: {resume_name}")
        except FileNotFoundError as exc:
            st.error(str(exc))
    else:
        uploaded = st.file_uploader(
            "Upload your resume (PDF, max 5 MB)", type=["pdf"]
        )
        if uploaded is not None:
            try:
                resume_text, resume_name = _resume_text_from_upload(uploaded)
                st.success(f"Extracted {len(resume_text)} characters from {resume_name}.")
            except ResumeParseError as exc:
                st.error(f"Could not read PDF: {exc}")

    jd_default = _load_sample_jd() if use_sample else ""
    jd_text = st.text_area(
        "Paste the job description",
        value=jd_default,
        height=200,
        placeholder="Paste the full job description here...",
    )

    return resume_text, resume_name, jd_text


def _render_score(composite) -> None:
    """Visual KPIs: composite ATS score with cosine + coverage breakdown."""
    col_main, col_bar = st.columns([1, 2])
    col_main.metric("ATS Match Score", f"{composite.composite_pct:.1f}%")
    col_bar.progress(
        min(composite.composite_pct / 100, 1.0),
        text=composite.verdict,
    )

    col_a, col_b = st.columns(2)
    col_a.metric("Cosine similarity (70%)", f"{composite.cosine_pct:.1f}%")
    col_b.metric("Keyword coverage (30%)", f"{composite.coverage_pct:.1f}%")

    st.caption(
        "Final score = 70% cosine similarity (vocabulary weighting) + "
        "30% keyword coverage (fraction of JD keywords your resume mentions)."
    )


def _render_skills(analysis) -> None:
    """Two-column matched / missing summary plus a small bar chart."""
    col_m, col_x = st.columns(2)
    with col_m:
        st.markdown("#### Matched keywords")
        if analysis.matched:
            st.write(", ".join(analysis.matched[:30]))
        else:
            st.write("None.")
    with col_x:
        st.markdown("#### Missing keywords")
        if analysis.missing_tech:
            st.markdown(
                "**Top tech gaps:** " + ", ".join(analysis.missing_tech[:10])
            )
        if analysis.missing:
            st.write(", ".join(analysis.missing[:30]))
        else:
            st.write("None.")

    counts = pd.DataFrame(
        {
            "count": [len(analysis.matched), len(analysis.missing)],
        },
        index=["matched", "missing"],
    )
    st.bar_chart(counts, height=200)


def _render_suggestions(analysis) -> None:
    st.markdown("#### Suggestions")
    for tip in analysis.suggestions:
        st.markdown(f"- {tip}")


def _render_history() -> None:
    history = load_history()
    if history.empty:
        return
    st.subheader("4. Analysis history")
    st.dataframe(
        history.sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def render() -> None:
    st.title("Resume ATS Analyzer")
    st.caption(
        "Upload a resume, paste a job description, and see your ATS match, "
        "keyword gaps, and improvement suggestions."
    )

    resume_text, resume_name, jd_text = _render_inputs()

    if not resume_text or not jd_text.strip():
        st.info("Provide both a resume and a job description to run the analysis.")
        _render_history()
        return

    run = st.button("Run analysis", type="primary")
    if not run:
        _render_history()
        return

    try:
        match = compute_match(resume_text, jd_text)
        analysis = analyse_keywords(resume_text, jd_text)
    except ValueError as exc:
        st.error(str(exc))
        return

    composite = compose_score(match.percent, analysis.coverage_pct)

    st.subheader("2. ATS score")
    _render_score(composite)

    st.subheader("3. Skills analysis")
    _render_skills(analysis)
    _render_suggestions(analysis)

    # Persist and offer download.
    saved = save_report(match, analysis, resume_name or "uploaded.pdf", jd_text, composite)
    frame = build_report_frame(match, analysis, resume_name or "uploaded.pdf", jd_text, composite)
    st.download_button(
        "Download CSV report",
        data=report_csv_bytes(frame),
        file_name=Path(saved["report_path"]).name,
        mime="text/csv",
    )

    _render_history()


if __name__ == "__main__":
    render()
else:
    render()
