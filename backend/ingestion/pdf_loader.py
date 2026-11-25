
import re
from pathlib import Path

import pymupdf


class InvalidPDFError(Exception):
    """Raised when a PDF cannot be read or has no extractable text."""


def clean_text(text: str) -> str:
    """Clean extracted PDF text while preserving paragraph structure."""

    text = re.sub(r"-\n(\w)", r"\1", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extract text page by page, preserving source and page metadata."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    # Check whether the uploaded file contains a PDF signature.
    with path.open("rb") as file:
        header = file.read(1024)

    if b"%PDF-" not in header:
        raise InvalidPDFError(
            "The uploaded file is not a valid PDF."
        )

    extracted_pages = []

    try:
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted:
                raise InvalidPDFError(
                    "Password-protected PDFs are not supported."
                )

            for page_number, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text("text"))

                if text:
                    extracted_pages.append({
                        "text": text,
                        "metadata": {
                            "source": path.name,
                            "page": page_number,
                        },
                    })

    except InvalidPDFError:
        raise
    except Exception as exc:
        raise InvalidPDFError(
            "The uploaded file is not a readable PDF."
        ) from exc

    if not extracted_pages:
        raise InvalidPDFError(
            "No extractable text found. The PDF may contain scanned images or empty pages."
        )

    return extracted_pages
