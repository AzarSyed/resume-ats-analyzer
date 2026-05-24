"""Resume PDF parsing.

We try ``pdfplumber`` first because it handles multi-column layouts well
(common in resume templates). If that yields nothing, we fall back to
``PyPDF2``. The function also accepts raw bytes so the Streamlit
``file_uploader`` widget can pass its uploads straight through without
hitting disk.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pdfplumber
from PyPDF2 import PdfReader

from utils.logger import get_logger
from utils.settings import MAX_PDF_BYTES

log = get_logger(__name__)

PdfSource = Union[str, Path, bytes, io.BytesIO]


class ResumeParseError(ValueError):
    """Raised when a PDF cannot be parsed into meaningful text."""


def _to_buffer(source: PdfSource) -> io.BytesIO:
    """Normalise the supported input types to an in-memory buffer."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as handle:
            data = handle.read()
    elif isinstance(source, bytes):
        data = source
    elif isinstance(source, io.BytesIO):
        data = source.getvalue()
    else:
        raise ResumeParseError(f"Unsupported PDF source type: {type(source)!r}")

    if len(data) == 0:
        raise ResumeParseError("PDF is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise ResumeParseError(
            f"PDF is too large ({len(data)} bytes, max {MAX_PDF_BYTES})."
        )
    return io.BytesIO(data)


def _extract_with_pdfplumber(buffer: io.BytesIO) -> str:
    buffer.seek(0)
    pages: list[str] = []
    with pdfplumber.open(buffer) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    return "\n".join(pages)


def _extract_with_pypdf2(buffer: io.BytesIO) -> str:
    buffer.seek(0)
    reader = PdfReader(buffer)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page for page in pages if page.strip())


def extract_text(source: PdfSource) -> str:
    """Return the readable text content of a PDF resume.

    Raises ``ResumeParseError`` when no text can be recovered (e.g. the
    PDF is a scanned image without OCR).
    """
    buffer = _to_buffer(source)

    try:
        text = _extract_with_pdfplumber(buffer)
        if text.strip():
            log.info("Extracted %d chars via pdfplumber.", len(text))
            return text
    except Exception as exc:
        log.warning("pdfplumber failed: %s; trying PyPDF2.", exc)

    try:
        text = _extract_with_pypdf2(buffer)
    except Exception as exc:
        raise ResumeParseError(f"PyPDF2 also failed: {exc}") from exc

    if not text.strip():
        raise ResumeParseError(
            "No text could be extracted. The PDF may be a scanned image; "
            "run it through OCR first."
        )

    log.info("Extracted %d chars via PyPDF2 fallback.", len(text))
    return text
