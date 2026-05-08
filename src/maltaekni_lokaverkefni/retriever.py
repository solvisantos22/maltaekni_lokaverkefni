"""Retrievers over processed retrieval chunks."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import Embeddings
from .types_classes import Chunk
from .ice_tokenizer import IceTokenizer


EMBEDDING_METHODS = {
    "icebert": "IceBert",
    "bge-m3": "BGE-M3",
}

RRF_METHODS = {
    "rrf-icebert-bm25": "icebert",
    "rrf-bge-m3-bm25": "bge-m3",
}

SUPPORTED_METHODS = {"tfidf", "bm25", *EMBEDDING_METHODS, *RRF_METHODS}


class Retriever:
    """
    Retriever for the MVP RAG pipeline.

    Explanation:
        Retriever indexes processed Chunk objects and returns the most relevant
        chunks for a user question. It supports lexical retrieval, embedding
        retrieval, and reciprocal rank fusion between BM25 and embeddings.

    Attributes:
        chunks: Chunk objects currently indexed by the retriever.
        method: Retrieval method name.
        ice_tokenizer: Icelandic tokenizer used before lexical and embedding retrieval.
        tokenized_chunks: Tokenized chunk texts used by BM25.
        bm25: BM25Okapi index when method uses BM25.
        embedder: Embedding encoder when method uses embeddings.
        embedding_matrix: Normalized embedding matrix for chunks.
        vectorizer: TfidfVectorizer when method is "tfidf".
        chunk_matrix: TF-IDF matrix when method is "tfidf".

    Public methods:
        fit(chunks): Build the retrieval index from Chunk objects.
        search(question, top_k=3): Return ranked chunks for a question.
    """

    def __init__(self, method, *, cache_dir: Path | None = None):
        """Initialize an unfitted retriever for a supported retrieval method."""
        method = self.__normalize_method(method)
        self.chunks: list[Chunk] = []
        self.method = method
        self.ice_tokenizer = IceTokenizer()
        self.tokenized_chunks: list[list[str]] = []
        self.bm25 = None
        self.embedder = None
        self.embedding_matrix = None
        self.cache_dir = cache_dir
        self.chunk_fingerprint = ""
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                analyzer=self.__analyze,
                smooth_idf=True
            )
            self.chunk_matrix = None
        else:
            self.vectorizer = None
            self.chunk_matrix = None

        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unknown retrieval method: {method}")

    def fit(self, chunks: list[Chunk]):
        """Fit the selected retrieval method on chunk text."""
        if not chunks:
            raise ValueError("Cannot fit retriever with no chunks")
        self.chunks = chunks
        texts = [self._searchable_text(chunk) for chunk in chunks]
        self.chunk_fingerprint = self.__fingerprint_chunks(chunks)
        if self.method == "tfidf":
            self.__fit_tfidf(texts)
        elif self.method == "bm25":
            self.__fit_bm25(texts)
        elif self.method in EMBEDDING_METHODS:
            self.__fit_embeddings(texts, self.method)
        elif self.method in RRF_METHODS:
            self.__fit_bm25(texts)
            self.__fit_embeddings(texts, RRF_METHODS[self.method])
        
    def __fit_tfidf(self, texts):
        """Fit the TF-IDF vectorizer on searchable chunk texts."""
        self.chunk_matrix = self.vectorizer.fit_transform(texts)

    def __fit_bm25(self, texts):
        """Fit the BM25 index on tokenized searchable chunk texts."""
        self.tokenized_chunks = [self.__analyze(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def __fit_embeddings(self, texts: list[str], method: str):
        """Fit or load chunk embeddings for an embedding retrieval method."""
        tokenized_texts = [self.__analyze(text) for text in texts]
        self.embedder = Embeddings(model=EMBEDDING_METHODS[method])

        cache_path = self.__embedding_cache_path(method)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as cached:
                if cached["fingerprint"].item() == self.chunk_fingerprint:
                    self.embedding_matrix = cached["embeddings"]
                    return

        self.embedding_matrix = self.embedder.fit(tokenized_texts)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                fingerprint=self.chunk_fingerprint,
                embeddings=self.embedding_matrix,
            )
        

    def search(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Return top chunks in the retrieval contract shape."""
        if self.method == "tfidf":
            scores = self.__search_tfidf(question)
        elif self.method == "bm25":
            scores = self.__search_bm25(question)
        elif self.method in EMBEDDING_METHODS:
            scores = self.__search_embeddings(question)
        elif self.method in RRF_METHODS:
            scores = self.__search_rrf(question)
        else:
            raise ValueError(f"Search is not implemented for method {self.method}")

        top_indexes = scores.argsort()[::-1][:top_k]

        return self.__build_result(question, top_indexes, scores)

    def __search_tfidf(self, question: str):
        """Score all chunks against a question with cosine similarity."""
        if self.chunk_matrix is None:
            raise ValueError("TF-IDF retriever must be fit before search")

        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.chunk_matrix).flatten()
        return scores

    def __search_bm25(self, question: str):
        """Score all chunks against a question with BM25."""
        if self.bm25 is None:
            raise ValueError("BM25 retriever must be fit before search")

        return self.bm25.get_scores(self.__analyze(question))

    def __search_embeddings(self, question: str):
        """Score all chunks against a question with embedding cosine similarity."""
        if self.embedder is None or self.embedding_matrix is None:
            raise ValueError("Embedding retriever must be fit before search")

        query_tokens = self.__analyze(question)
        query_embedding = self.embedder.transform([query_tokens])[0]
        return self.embedding_matrix @ query_embedding

    def __search_rrf(self, question: str, rank_constant: int = 60):
        """Combine BM25 and embedding ranks with reciprocal rank fusion."""
        bm25_scores = self.__search_bm25(question)
        embedding_scores = self.__search_embeddings(question)
        return self.__rrf_scores([bm25_scores, embedding_scores], rank_constant)

    def __build_result(self, question: str, top_indexes, scores) -> dict[str, Any]:
        """Build the retrieval contract response from ranked chunk indexes."""
        result_chunks = []
        for index in top_indexes:
            score = float(scores[index])
            if score <= 0:
                continue

            chunk_dict = asdict(self.chunks[index])
            chunk_dict["score"] = score
            chunk_dict["retrieval_method"] = self.method
            result_chunks.append(chunk_dict)

        return {
            "question": question,
            "chunks": result_chunks,
        }

    def __embedding_cache_path(self, method: str) -> Path | None:
        """Return the local cache path for chunk embeddings."""
        if self.cache_dir is None:
            return None

        return self.cache_dir / f"{method}-{self.chunk_fingerprint}.npz"

    def __fingerprint_chunks(self, chunks: list[Chunk]) -> str:
        """Build a stable fingerprint for cache invalidation."""
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._searchable_text(chunk).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    def __rrf_scores(self, ranked_score_lists, rank_constant: int):
        """Convert score lists to reciprocal-rank-fusion scores."""
        fused = np.zeros(len(self.chunks), dtype=float)
        for scores in ranked_score_lists:
            ranked_indexes = np.argsort(scores)[::-1]
            for rank, index in enumerate(ranked_indexes, start=1):
                fused[index] += 1.0 / (rank_constant + rank)
        return fused

    def __normalize_method(self, method: str) -> str:
        """Normalize user-facing method aliases."""
        aliases = {
            "IceBert": "icebert",
            "IceBERT": "icebert",
            "BGE-M3": "bge-m3",
            "bgem3": "bge-m3",
            "bgm3": "bge-m3",
            "hybrid-icebert": "rrf-icebert-bm25",
            "hybrid-bge-m3": "rrf-bge-m3-bm25",
        }
        return aliases.get(method, method)

    def _searchable_text(self, chunk: Chunk) -> str:
        """Weight titles/sections a little more than body text."""
        return "\n".join(
            [
                chunk.title,
                chunk.section,
                chunk.section,
                chunk.section,
                chunk.text,
            ]
        )

    def __analyze(self, text: str) -> list[str]:
        """Analyze text into lowercase Icelandic tokens for retrieval."""
        
        tokens = self.ice_tokenizer.tokenIce(text)

        return [
            token
            for token in tokens
            if len(token) > 1 and re.search(r"[\wáðéíóúýþæö]", token)
        ]

    


def load_chunks(path: Path) -> list[Chunk]:
    """Load chunks from data/processed/chunks.json."""
    with path.open("r", encoding="utf-8") as file:
        raw_chunks = json.load(file)

    return [
        Chunk(
            chunk_id=raw_chunk["chunk_id"],
            text=raw_chunk["text"],
            source=raw_chunk["source"],
            title=raw_chunk["title"],
            url=raw_chunk["url"],
            section=raw_chunk["section"],
        )
        for raw_chunk in raw_chunks
    ]


def build_retriever(method, chunks_path: Path = Path("data/processed/chunks.json")) -> Retriever:
    """Load chunks and return a fitted retriever."""
    chunks = load_chunks(chunks_path)
    retriever = Retriever(method, cache_dir=chunks_path.parent / "embedding_cache")
    retriever.fit(chunks)
    return retriever


if __name__ == "__main__":
    question = "Hvað get ég gert ef vara er gölluð?"
    for method in ["tfidf", "bm25"]:
        retriever = build_retriever(method)
        result = retriever.search(question)
        print(json.dumps(result, ensure_ascii=False, indent=2))
