
import pytest
import pymupdf

from backend.ingestion.pdf_loader import (
    InvalidPDFError,
    clean_text,
    extract_pdf_pages,
)
from backend.ingestion.text_splitter import (
    split_text,
    split_pages_into_chunks,
)


def create_test_pdf(path, pages):
    """Create a temporary PDF with the supplied page contents."""

    document = pymupdf.open()

    for content in pages:
        page = document.new_page()

        if content:
            page.insert_text((72, 72), content)

    document.save(path)
    document.close()

    return path


# --------------------------------------------------
# PDF LOADER TESTS
# --------------------------------------------------

def test_clean_text_normalizes_whitespace():
    text = "Hello    world\r\n\r\n\r\nThis is a test."

    result = clean_text(text)

    assert result == "Hello world\n\nThis is a test."


def test_clean_text_joins_hyphenated_words():
    text = "inter-\nnal processing"

    result = clean_text(text)

    assert result == "internal processing"


def test_extract_pdf_pages_preserves_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    create_test_pdf(
        pdf_path,
        [
            "First page content",
            "",
            "Third page content",
        ],
    )

    pages = extract_pdf_pages(str(pdf_path))

    assert len(pages) == 2

    assert pages[0]["metadata"] == {
        "source": "sample.pdf",
        "page": 1,
    }

    assert pages[1]["metadata"] == {
        "source": "sample.pdf",
        "page": 3,
    }

    assert "First page content" in pages[0]["text"]
    assert "Third page content" in pages[1]["text"]


def test_extract_pdf_pages_rejects_missing_file(tmp_path):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(str(missing_file))


def test_extract_pdf_pages_rejects_non_pdf_extension(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("This is not a PDF.")

    with pytest.raises(ValueError, match="Only PDF files"):
        extract_pdf_pages(str(file_path))


def test_extract_pdf_pages_rejects_invalid_pdf_signature(tmp_path):
    file_path = tmp_path / "invalid.pdf"
    file_path.write_bytes(b"This is not a valid PDF file.")

    with pytest.raises(InvalidPDFError):
        extract_pdf_pages(str(file_path))


def test_extract_pdf_pages_rejects_empty_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    create_test_pdf(pdf_path, ["", ""])

    with pytest.raises(InvalidPDFError, match="No extractable text"):
        extract_pdf_pages(str(pdf_path))


# --------------------------------------------------
# TEXT SPLITTER TESTS
# --------------------------------------------------

def test_split_text_returns_empty_list_for_empty_input():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_respects_chunk_size():
    text = "abcdefghijklmnopqrstuvwxyz" * 5

    chunks = split_text(
        text,
        chunk_size=30,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_split_text_preserves_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 3

    chunks = split_text(
        text,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert chunks[0][-5:] == chunks[1][:5]


def test_split_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("Some text", chunk_size=0)


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [
        (20, -1),
        (20, 20),
        (20, 25),
    ],
)
def test_split_text_rejects_invalid_overlap(
    chunk_size,
    chunk_overlap,
):
    with pytest.raises(ValueError):
        split_text(
            "Some text",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


# --------------------------------------------------
# PAGE-TO-CHUNK TESTS
# --------------------------------------------------

def test_split_pages_preserves_metadata():
    pages = [
        {
            "text": "A" * 100,
            "metadata": {
                "source": "sample.pdf",
                "page": 1,
            },
        },
        {
            "text": "B" * 100,
            "metadata": {
                "source": "sample.pdf",
                "page": 2,
            },
        },
    ]

    chunks = split_pages_into_chunks(
        pages,
        chunk_size=40,
        chunk_overlap=10,
    )

    assert len(chunks) > 2

    assert all(
        chunk["metadata"]["source"] == "sample.pdf"
        for chunk in chunks
    )

    assert {chunk["metadata"]["page"] for chunk in chunks} == {1, 2}

    for chunk in chunks:
        assert "chunk_index" in chunk["metadata"]
        assert len(chunk["text"]) <= 40


def test_split_pages_resets_chunk_index_per_page():
    pages = [
        {
            "text": "A" * 100,
            "metadata": {
                "source": "sample.pdf",
                "page": 1,
            },
        },
        {
            "text": "B" * 100,
            "metadata": {
                "source": "sample.pdf",
                "page": 2,
            },
        },
    ]

    chunks = split_pages_into_chunks(
        pages,
        chunk_size=40,
        chunk_overlap=10,
    )

    page_one_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"]["page"] == 1
    ]

    page_two_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"]["page"] == 2
    ]

    assert page_one_chunks[0]["metadata"]["chunk_index"] == 0
    assert page_two_chunks[0]["metadata"]["chunk_index"] == 0
