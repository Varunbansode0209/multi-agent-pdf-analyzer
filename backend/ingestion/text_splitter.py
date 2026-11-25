
from backend.core.config import settings


SEPARATORS = ["\n\n", "\n", ". ", " "]


def split_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        if end < text_length:

            min_end = start + chunk_size // 2

            # Prefer paragraph, newline, sentence, then word boundaries.
            for separator in SEPARATORS:

                position = text.rfind(separator, start, end)

                if position > min_end:
                    end = position + len(separator)
                    break

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(0, end - chunk_overlap)

        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks


def split_pages_into_chunks(
    pages: list[dict],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[dict]:

    all_chunks = []

    for page in pages:

        page_chunks = split_text(
            text=page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(page_chunks):

            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    **page["metadata"],
                    "chunk_index": chunk_index,
                },
            })

    return all_chunks
