"""Embedding helpers for pre-tokenized retrieval chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


EmbeddingModelName = Literal["BGE-M3", "IceBert"]


MODEL_IDS: dict[str, str] = {
    "BGE-M3": "BAAI/bge-m3",
    "IceBert": "mideind/IceBERT",
}


DEFAULT_MAX_LENGTHS: dict[str, int] = {
    "BGE-M3": 8192,
    "IceBert": 512,
}


class Embeddings:
    """
    Encode pre-tokenized chunks with either BGE-M3 or IceBert.

    Explanation:
        These models are already pretrained, so fit() does not train model
        weights. It encodes the provided tokenized chunks and stores the
        resulting normalized embedding matrix for later retrieval use.

    Attributes:
        model_name: Friendly model name, either "BGE-M3" or "IceBert".
        model_id: Hugging Face model id used to load tokenizer/model weights.
        chunk_embeddings: Matrix produced by fit(), one row per chunk.

    Public methods:
        fit(tokenized_chunks): Encode and store chunk embeddings.
        transform(tokenized_chunks): Encode tokenized chunks without storing.
    """

    def __init__(
        self,
        model: EmbeddingModelName | str = "BGE-M3",
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int | None = None,
    ):
        """Initialize an embedding encoder for a supported model."""
        if model not in MODEL_IDS:
            supported = ", ".join(MODEL_IDS)
            raise ValueError(f"Unknown embedding model: {model}. Use one of: {supported}")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = model
        self.model_id = MODEL_IDS[model]
        self.batch_size = batch_size
        self.max_length = max_length or DEFAULT_MAX_LENGTHS[model]
        self.chunk_embeddings = None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def fit(self, tokenized_chunks: Sequence[Sequence[str]]):
        """Encode and store embeddings for already-tokenized chunks."""
        self.chunk_embeddings = self.transform(tokenized_chunks)
        return self.chunk_embeddings

    def transform(self, tokenized_chunks: Sequence[Sequence[str]]):
        """Encode already-tokenized chunks into L2-normalized embeddings."""
        self.__validate_tokenized_chunks(tokenized_chunks)
        vectors = []

        for start in range(0, len(tokenized_chunks), self.batch_size):
            batch = [list(tokens) for tokens in tokenized_chunks[start:start + self.batch_size]]
            encoded = self.tokenizer(
                batch,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            pooled = self.__mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors.append(functional.normalize(pooled, p=2, dim=1).cpu().numpy())

        return np.vstack(vectors)

    def __mean_pool(self, token_embeddings: Any, attention_mask: Any):
        """Mean-pool token embeddings while ignoring padding tokens."""
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def __validate_tokenized_chunks(self, tokenized_chunks: Sequence[Sequence[str]]):
        """Validate the expected list-of-token-lists input shape."""
        if not tokenized_chunks:
            raise ValueError("Cannot fit embeddings with no chunks")

        for index, tokens in enumerate(tokenized_chunks):
            if isinstance(tokens, str) or not isinstance(tokens, Sequence):
                raise TypeError(
                    "Each chunk must be a sequence of tokens, not raw text. "
                    f"Chunk {index} has type {type(tokens).__name__}."
                )

            if not tokens:
                raise ValueError(f"Chunk {index} has no tokens")

            if not all(isinstance(token, str) for token in tokens):
                raise TypeError(f"Chunk {index} contains a non-string token")


if __name__ == "__main__":
    embedder = Embeddings(model="BGE-M3")
    from ice_tokenizer import IceTokenizer
    it = IceTokenizer()
    tokens = it.tokenIce("Hvað getur neytandi gert ef vara er gölluð")
    print(embedder.fit([tokens]))