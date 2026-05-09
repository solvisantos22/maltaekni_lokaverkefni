"""Cross-encoder reranking helpers for retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RerankerModelName = Literal["BGE-Reranker-v2-m3"]


MODEL_IDS: dict[str, str] = {
    "BGE-Reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}


DEFAULT_MAX_LENGTHS: dict[str, int] = {
    "BGE-Reranker-v2-m3": 1024,
}


class Reranker:
    """
    Score query-document pairs with a multilingual cross-encoder reranker.

    Explanation:
        Reranker is used after a first-stage retriever has selected candidate
        chunks. It reads the question and candidate chunk text together, then
        returns a relevance score for each candidate.

    Attributes:
        model_name: Friendly reranker model name.
        model_id: Hugging Face model id used to load tokenizer/model weights.
        device: Torch device used for inference.
        batch_size: Number of query-document pairs scored per batch.
        max_length: Maximum pair token length for the reranker model.

    Public methods:
        score(question, documents): Return one relevance score per document.
    """

    def __init__(
        self,
        model: RerankerModelName | str = "BGE-Reranker-v2-m3",
        *,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int | None = None,
    ):
        """Initialize a supported reranker model."""
        if model not in MODEL_IDS:
            supported = ", ".join(MODEL_IDS)
            raise ValueError(f"Unknown reranker model: {model}. Use one of: {supported}")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = model
        self.model_id = MODEL_IDS[model]
        self.batch_size = batch_size
        self.max_length = max_length or DEFAULT_MAX_LENGTHS[model]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(
            self.device
        )
        self.model.eval()

    def score(self, question: str, documents: Sequence[str]) -> np.ndarray:
        """Return normalized relevance scores for candidate documents."""
        self.__validate_inputs(question, documents)
        scores = []

        for start in range(0, len(documents), self.batch_size):
            batch_documents = list(documents[start:start + self.batch_size])
            encoded = self.tokenizer(
                [question] * len(batch_documents),
                batch_documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits

            batch_scores = logits.view(-1).float().sigmoid().cpu().numpy()
            scores.append(batch_scores)

        return np.concatenate(scores)

    def __validate_inputs(self, question: str, documents: Sequence[str]) -> None:
        """Validate reranker inputs before model inference."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        if not documents:
            raise ValueError("documents must contain at least one candidate")

        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise TypeError(
                    "Each reranker document must be raw text. "
                    f"Item {index} has type {type(document).__name__}."
                )

            if not document.strip():
                raise ValueError(f"Reranker document {index} is empty")
