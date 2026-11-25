
from backend.core.config import settings


class EmbeddingGenerator:

    def __init__(self):
        self._model = None
        self._client = None

    @property
    def model(self):
        """Load local embedding model only when needed."""

        if self._model is None:

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL
            )

        return self._model

    @property
    def client(self):
        """Initialize OpenAI client only when needed."""

        if self._client is None:

            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_MAX_RETRIES,
            )

        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:

        if not texts:
            return []

        backend = settings.EMBEDDING_BACKEND.lower()

        if backend == "local":

            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return vectors.tolist()

        elif backend == "openai":

            vectors = []

            for i in range(0, len(texts), 100):

                batch = texts[i:i + 100]

                response = self.client.embeddings.create(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    input=batch,
                )

                vectors.extend(
                    item.embedding for item in response.data
                )

            return vectors

        else:
            raise ValueError(
                f"Unsupported embedding backend: {backend}"
            )

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        return self._embed(texts)

    def generate_query_embedding(
        self,
        query: str
    ) -> list[float]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self._embed([query])[0]


embedding_generator = EmbeddingGenerator()
