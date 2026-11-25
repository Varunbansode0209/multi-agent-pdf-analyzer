
import pytest

from backend.core.config import settings
from backend.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Create an isolated ChromaDB instance for each test."""

    database_path = tmp_path / "test_chroma"

    monkeypatch.setattr(
        settings,
        "CHROMA_PERSIST_DIR",
        str(database_path),
    )

    return ChromaVectorStore()


def add_sample_documents(store):
    """Insert sample documents with manually generated embeddings."""

    store.add_documents(
        documents=[
            "Python is a programming language.",
            "Machine learning uses data to train models.",
            "Deep learning uses neural networks.",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        metadatas=[
            {
                "source": "python.pdf",
                "page": 1,
                "chunk_index": 0,
            },
            {
                "source": "ml.pdf",
                "page": 1,
                "chunk_index": 0,
            },
            {
                "source": "ml.pdf",
                "page": 2,
                "chunk_index": 0,
            },
        ],
        ids=["doc1", "doc2", "doc3"],
    )


# --------------------------------------------------
# VECTOR STORE INITIALIZATION
# --------------------------------------------------

def test_vector_store_initializes(store):
    assert store.collection is not None
    assert store.count() == 0


def test_empty_collection_returns_zero(store):
    assert store.count() == 0


def test_empty_collection_returns_no_search_results(store):
    results = store.search([1.0, 0.0, 0.0])

    assert results == []


# --------------------------------------------------
# DOCUMENT INSERTION
# --------------------------------------------------

def test_add_documents_successfully(store):
    add_sample_documents(store)

    assert store.count() == 3


def test_add_empty_documents_does_nothing(store):
    store.add_documents(
        documents=[],
        embeddings=[],
        metadatas=[],
        ids=[],
    )

    assert store.count() == 0


def test_add_documents_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError, match="equal lengths"):
        store.add_documents(
            documents=["Document one", "Document two"],
            embeddings=[[1.0, 0.0]],
            metadatas=[{"source": "test.pdf"}],
            ids=["doc1"],
        )


def test_add_documents_rejects_different_embedding_dimensions(store):
    with pytest.raises(ValueError, match="same dimensions"):
        store.add_documents(
            documents=["Document one", "Document two"],
            embeddings=[
                [1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            metadatas=[
                {"source": "test.pdf"},
                {"source": "test.pdf"},
            ],
            ids=["doc1", "doc2"],
        )


def test_upsert_replaces_existing_document(store):
    store.add_documents(
        documents=["Original content"],
        embeddings=[[1.0, 0.0]],
        metadatas=[
            {
                "source": "test.pdf",
                "page": 1,
                "chunk_index": 0,
            }
        ],
        ids=["doc1"],
    )

    store.add_documents(
        documents=["Updated content"],
        embeddings=[[1.0, 0.0]],
        metadatas=[
            {
                "source": "test.pdf",
                "page": 1,
                "chunk_index": 0,
            }
        ],
        ids=["doc1"],
    )

    assert store.count() == 1

    chunks = store.get_all_chunks()

    assert chunks[0]["text"] == "Updated content"


# --------------------------------------------------
# SIMILARITY CALCULATION
# --------------------------------------------------

@pytest.mark.parametrize(
    "distance, expected",
    [
        (0.0, 1.0),
        (0.2, 0.8),
        (0.5, 0.5),
        (1.0, 0.0),
        (1.5, 0.0),
        (-0.5, 1.0),
    ],
)
def test_to_similarity(distance, expected):
    result = ChromaVectorStore.to_similarity(distance)

    assert result == pytest.approx(expected)


# --------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------

def test_search_returns_relevant_document(store):
    add_sample_documents(store)

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=3,
    )

    assert len(results) == 3

    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "Python is a programming language."
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_search_respects_top_k(store):
    add_sample_documents(store)

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2


def test_search_filters_by_source(store):
    add_sample_documents(store)

    results = store.search(
        query_embedding=[0.0, 1.0, 0.0],
        top_k=3,
        where={"source": "ml.pdf"},
    )

    assert len(results) == 2

    assert all(
        item["metadata"]["source"] == "ml.pdf"
        for item in results
    )


def test_search_rejects_empty_query_embedding(store):
    with pytest.raises(ValueError, match="cannot be empty"):
        store.search([])


def test_search_rejects_invalid_top_k(store):
    with pytest.raises(ValueError, match="greater than zero"):
        store.search([1.0, 0.0], top_k=0)


# --------------------------------------------------
# CHUNK RETRIEVAL AND SORTING
# --------------------------------------------------

def test_get_all_chunks_returns_sorted_chunks(store):
    store.add_documents(
        documents=[
            "Page two content",
            "Page one content",
            "Another page one chunk",
        ],
        embeddings=[
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.2],
        ],
        metadatas=[
            {
                "source": "test.pdf",
                "page": 2,
                "chunk_index": 0,
            },
            {
                "source": "test.pdf",
                "page": 1,
                "chunk_index": 1,
            },
            {
                "source": "test.pdf",
                "page": 1,
                "chunk_index": 0,
            },
        ],
        ids=["doc3", "doc2", "doc1"],
    )

    chunks = store.get_all_chunks()

    assert [chunk["id"] for chunk in chunks] == [
        "doc1",
        "doc2",
        "doc3",
    ]


def test_get_all_chunks_filters_by_source(store):
    add_sample_documents(store)

    chunks = store.get_all_chunks(source="ml.pdf")

    assert len(chunks) == 2

    assert all(
        chunk["metadata"]["source"] == "ml.pdf"
        for chunk in chunks
    )


# --------------------------------------------------
# DOCUMENT MANAGEMENT
# --------------------------------------------------

def test_list_sources_returns_document_statistics(store):
    add_sample_documents(store)

    sources = store.list_sources()

    source_map = {
        item["source"]: item
        for item in sources
    }

    assert source_map["python.pdf"]["chunks"] == 1
    assert source_map["python.pdf"]["pages"] == 1

    assert source_map["ml.pdf"]["chunks"] == 2
    assert source_map["ml.pdf"]["pages"] == 2


def test_delete_by_source_removes_document_chunks(store):
    add_sample_documents(store)

    store.delete_by_source("ml.pdf")

    assert store.count() == 1

    remaining = store.get_all_chunks()

    assert remaining[0]["metadata"]["source"] == "python.pdf"


def test_delete_by_source_does_not_delete_other_documents(store):
    add_sample_documents(store)

    store.delete_by_source("python.pdf")

    assert store.count() == 2

    sources = store.list_sources()

    assert len(sources) == 1
    assert sources[0]["source"] == "ml.pdf"
