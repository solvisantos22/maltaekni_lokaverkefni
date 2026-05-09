"""Embedding helpers for raw retrieval text."""

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
    Encode retrieval text with either BGE-M3 or IceBert.

    Explanation:
        These models are already pretrained, so fit() does not train model
        weights. It encodes the provided raw chunk text and stores the resulting
        normalized embedding matrix for later retrieval use.

    Attributes:
        model_name: Friendly model name, either "BGE-M3" or "IceBert".
        model_id: Hugging Face model id used to load tokenizer/model weights.
        chunk_embeddings: Matrix produced by fit(), one row per chunk.

    Public methods:
        fit(texts): Encode and store chunk embeddings.
        transform(texts): Encode texts without storing.
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

    def fit(self, texts: Sequence[str]):
        """Encode and store embeddings for raw chunk texts."""
        self.chunk_embeddings = self.transform(texts, input_type="document")
        return self.chunk_embeddings

    def transform(self, texts: Sequence[str], *, input_type: str = "document"):
        """Encode raw texts into L2-normalized embeddings."""
        self.__validate_texts(texts)
        vectors = []

        for start in range(0, len(texts), self.batch_size):
            batch = [
                self.__format_text(text, input_type=input_type)
                for text in texts[start:start + self.batch_size]
            ]
            encoded = self.tokenizer(
                batch,
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

    def __format_text(self, text: str, *, input_type: str) -> str:
        """Apply model-specific query formatting where useful."""
        if self.model_name == "BGE-M3" and input_type == "query":
            return f"Represent this sentence for searching relevant passages: {text}"

        return text

    def __mean_pool(self, token_embeddings: Any, attention_mask: Any):
        """Mean-pool token embeddings while ignoring padding tokens."""
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def __validate_texts(self, texts: Sequence[str]):
        """Validate the expected raw-text input shape."""
        if not texts:
            raise ValueError("Cannot fit embeddings with no texts")

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    "Each embedding input must be raw text. "
                    f"Item {index} has type {type(text).__name__}."
                )

            if not text.strip():
                raise ValueError(f"Embedding input {index} is empty")


if __name__ == "__main__":
    embedder = Embeddings(model="BGE-M3")
    print(embedder.fit(["Hvað getur neytandi gert ef vara er gölluð"]))
