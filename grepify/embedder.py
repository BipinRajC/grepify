"""Embedding generation using sentence-transformers (E5 family).

E5 models require prefixes:
  - "passage: " for documents being indexed
  - "query: "   for search queries
Without prefixes, quality drops ~5-10%.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import Chunk


class Embedder:
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str = "cuda",
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed documents for indexing."""
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        return self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed a list of Chunk objects."""
        return self.embed_passages([c.text for c in chunks], batch_size)
