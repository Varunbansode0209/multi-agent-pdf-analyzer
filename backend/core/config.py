
import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")


class Settings:

    # --------------------------------------------------
    # OpenAI / LLM settings
    # --------------------------------------------------

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-6-luna")

    LLM_REASONING_EFFORT: str = os.getenv(
        "LLM_REASONING_EFFORT", "low"
    )

    LLM_TEMPERATURE: float = float(
        os.getenv("LLM_TEMPERATURE", "0.1")
    )

    LLM_TIMEOUT: int = int(
        os.getenv("LLM_TIMEOUT", "60")
    )

    LLM_MAX_RETRIES: int = int(
        os.getenv("LLM_MAX_RETRIES", "2")
    )

    # --------------------------------------------------
    # Embedding settings
    # --------------------------------------------------

    EMBEDDING_BACKEND: str = os.getenv(
        "EMBEDDING_BACKEND", "local"
    ).strip().lower()

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small"
    )

    # --------------------------------------------------
    # Document processing settings
    # --------------------------------------------------

    CHUNK_SIZE: int = int(
        os.getenv("CHUNK_SIZE", "1000")
    )

    CHUNK_OVERLAP: int = int(
        os.getenv("CHUNK_OVERLAP", "200")
    )

    TOP_K: int = int(
        os.getenv("TOP_K", "4")
    )

    MIN_SIMILARITY: float = float(
        os.getenv("MIN_SIMILARITY", "0.25")
    )

    MAX_CONTEXT_CHARS: int = int(
        os.getenv("MAX_CONTEXT_CHARS", "6000")
    )

    MAX_HISTORY_TURNS: int = int(
        os.getenv("MAX_HISTORY_TURNS", "3")
    )

    # --------------------------------------------------
    # Summarization settings
    # --------------------------------------------------

    MAP_BATCH_CHARS: int = int(
        os.getenv("MAP_BATCH_CHARS", "12000")
    )

    MAP_CONCURRENCY: int = int(
        os.getenv("MAP_CONCURRENCY", "5")
    )

    REDUCE_MAX_CHARS: int = int(
        os.getenv("REDUCE_MAX_CHARS", "16000")
    )

    # --------------------------------------------------
    # ChromaDB settings
    # --------------------------------------------------

    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        str(BASE_DIR / "data" / "vectorstore")
    )

    COLLECTION_NAME: str = os.getenv(
        "COLLECTION_NAME", "pdf_chunks"
    )

    @property
    def ACTIVE_COLLECTION_NAME(self) -> str:

        if self.EMBEDDING_BACKEND == "local":
            model_id = self.EMBEDDING_MODEL.split("/")[-1]

        elif self.EMBEDDING_BACKEND == "openai":
            model_id = self.OPENAI_EMBEDDING_MODEL

        else:
            raise ValueError(
                f"Unsupported embedding backend: {self.EMBEDDING_BACKEND}"
            )

        safe_model_id = "".join(
            char.lower() if char.isalnum() else "_"
            for char in model_id
        ).strip("_")

        return (
            f"{self.COLLECTION_NAME}_"
            f"{self.EMBEDDING_BACKEND}_"
            f"{safe_model_id}"
        )

    # --------------------------------------------------
    # File and upload settings
    # --------------------------------------------------

    UPLOAD_DIR: str = os.getenv(
        "UPLOAD_DIR",
        str(BASE_DIR / "data" / "uploads")
    )

    MAX_UPLOAD_MB: int = int(
        os.getenv("MAX_UPLOAD_MB", "20")
    )

    # --------------------------------------------------
    # FastAPI settings
    # --------------------------------------------------

    API_URL: str = os.getenv(
        "API_URL", "http://localhost:8000"
    )

    # --------------------------------------------------
    # Logging settings
    # --------------------------------------------------

    LOG_DIR: str = os.getenv(
        "LOG_DIR",
        str(BASE_DIR / "logs")
    )

    LOG_LEVEL: str = os.getenv(
        "LOG_LEVEL", "INFO"
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate_embedding_backend(self) -> None:

        if self.EMBEDDING_BACKEND not in {"local", "openai"}:
            raise ValueError(
                "EMBEDDING_BACKEND must be 'local' or 'openai'."
            )

        if (
            self.EMBEDDING_BACKEND == "openai"
            and not self.OPENAI_API_KEY
        ):
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI embeddings."
            )


    def validate(self) -> None:
        """Validate essential application configuration."""

        if not self.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        if not self.LLM_MODEL.strip():
            raise ValueError("LLM_MODEL cannot be empty.")

        self.validate_embedding_backend()

        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero.")

        if not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE:
            raise ValueError(
                "CHUNK_OVERLAP must be non-negative "
                "and smaller than CHUNK_SIZE."
            )

        if self.TOP_K <= 0:
            raise ValueError("TOP_K must be greater than zero.")

        if not 0 <= self.MIN_SIMILARITY <= 1:
            raise ValueError("MIN_SIMILARITY must be between 0 and 1.")

        if self.MAX_CONTEXT_CHARS <= 0:
            raise ValueError("MAX_CONTEXT_CHARS must be greater than zero.")

        if self.MAX_HISTORY_TURNS <= 0:
            raise ValueError("MAX_HISTORY_TURNS must be greater than zero.")

        if self.MAP_BATCH_CHARS <= 0:
            raise ValueError("MAP_BATCH_CHARS must be greater than zero.")

        if self.MAP_CONCURRENCY <= 0:
            raise ValueError("MAP_CONCURRENCY must be greater than zero.")

        if self.REDUCE_MAX_CHARS <= 0:
            raise ValueError("REDUCE_MAX_CHARS must be greater than zero.")

        if self.MAX_UPLOAD_MB <= 0:
            raise ValueError("MAX_UPLOAD_MB must be greater than zero.")

        if self.LLM_TIMEOUT <= 0:
            raise ValueError("LLM_TIMEOUT must be greater than zero.")

        if self.LLM_MAX_RETRIES < 0:
            raise ValueError("LLM_MAX_RETRIES cannot be negative.")

    # --------------------------------------------------
    # Directory initialization
    # --------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create required application directories."""

        for path in (
            self.CHROMA_PERSIST_DIR,
            self.UPLOAD_DIR,
            self.LOG_DIR,
        ):
            Path(path).mkdir(
                parents=True,
                exist_ok=True
            )


settings = Settings()
