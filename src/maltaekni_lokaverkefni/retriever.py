"""Retrieval backends used by the RAG app and evaluation pipeline.

The retriever is intentionally method-switchable so the report can compare
lexical search, embedding search, reciprocal rank fusion, and reranking under
the same input/output contract. Every method returns the same ranked chunk
shape, which keeps answer generation, the UI, and evaluation independent from
the retrieval implementation.
"""

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
from .reranker import Reranker
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

RERANK_METHODS = {
    "rrf-bge-m3-bm25-rerank": "bge-m3",
}

SUPPORTED_METHODS = {
    "tfidf",
    "bm25",
    *EMBEDDING_METHODS,
    *RRF_METHODS,
    *RERANK_METHODS,
}
DEFAULT_STOP_WORDS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "all_stop_words.txt"
)
DEFAULT_LEMMA_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "chunk_lemmas.json"
)


class Retriever:
    """
    Retriever for the MVP RAG pipeline.

    Explanation:
        Retriever indexes processed Chunk objects and returns the most relevant
        chunks for a user question. It supports TF-IDF, BM25, direct embedding
        retrieval, reciprocal rank fusion between BM25 and embeddings, and an
        optional reranker after first-stage retrieval.

    Attributes:
        chunks: Chunk objects currently indexed by the retriever.
        method: Retrieval method name.
        ice_tokenizer: Icelandic tokenizer used before lexical retrieval.
        tokenized_chunks: Tokenized chunk texts used by BM25.
        bm25: BM25Okapi index when method uses BM25.
        embedder: Embedding encoder when method uses embeddings.
        embedding_matrix: Normalized embedding matrix for chunks.
        reranker: Cross-encoder reranker when using a rerank method.
        vectorizer: TfidfVectorizer when method is "tfidf".
        chunk_matrix: TF-IDF matrix when method is "tfidf".

    Public methods:
        fit(chunks): Build the retrieval index from Chunk objects.
        search(question, top_k=3): Return ranked chunks for a question.
    """

    def __init__(
        self,
        method,
        *,
        cache_dir: Path | None = None,
        stop_words_path: Path | None = DEFAULT_STOP_WORDS_PATH,
        lemma_cache_path: Path | None = DEFAULT_LEMMA_CACHE_PATH,
        rrf_candidate_k: int = 50,
    ):
        """Initialize an unfitted retriever for a supported retrieval method."""
        if rrf_candidate_k < 1:
            raise ValueError("rrf_candidate_k must be at least 1")

        self.chunks: list[Chunk] = []
        self.method = method
        self.rrf_candidate_k = rrf_candidate_k
        self.ice_tokenizer = IceTokenizer()
        self.stop_words = self.__load_stop_words(stop_words_path)
        self.lemma_cache = self.__load_lemma_cache(lemma_cache_path)
        self.tokenized_chunks: list[list[str]] = []
        self.embedding_texts: list[str] = []
        self.bm25 = None
        self.embedder = None
        self.embedding_matrix = None
        self.reranker = None
        self.cache_dir = cache_dir
        self.chunk_fingerprint = ""
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                analyzer=str.split,
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
        self.embedding_texts = texts
        self.chunk_fingerprint = self.__fingerprint_chunks(chunks)
        if self.method == "tfidf":
            self.tokenized_chunks = self.__tokenize_chunks(chunks, texts)
            self.__fit_tfidf()
        elif self.method == "bm25":
            self.tokenized_chunks = self.__tokenize_chunks(chunks, texts)
            self.__fit_bm25()
        elif self.method in EMBEDDING_METHODS:
            self.__fit_embeddings(self.method)
        elif self.method in RRF_METHODS:
            self.tokenized_chunks = self.__tokenize_chunks(chunks, texts)
            self.__fit_bm25()
            self.__fit_embeddings(RRF_METHODS[self.method])
        elif self.method in RERANK_METHODS:
            self.tokenized_chunks = self.__tokenize_chunks(chunks, texts)
            self.__fit_bm25()
            self.__fit_embeddings(RERANK_METHODS[self.method])
            self.__fit_reranker()
        
    def __fit_tfidf(self):
        """Fit the TF-IDF vectorizer on cached/analyzed chunk tokens."""
        self.chunk_matrix = self.vectorizer.fit_transform(
            [" ".join(tokens) for tokens in self.tokenized_chunks]
        )

    def __fit_bm25(self):
        """Fit the BM25 index on tokenized searchable chunk texts."""
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def __fit_embeddings(self, method: str):
        """Fit or load chunk embeddings for an embedding retrieval method."""
        self.embedder = Embeddings(model=EMBEDDING_METHODS[method])

        cache_path = self.__embedding_cache_path(method)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as cached:
                if cached["fingerprint"].item() == self.chunk_fingerprint:
                    self.embedding_matrix = cached["embeddings"]
                    return

        self.embedding_matrix = self.embedder.fit(self.embedding_texts)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                fingerprint=self.chunk_fingerprint,
                embeddings=self.embedding_matrix,
            )

    def __fit_reranker(self):
        """Load the cross-encoder reranker used after first-stage retrieval."""
        self.reranker = Reranker(model="BGE-Reranker-v2-m3")
        

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
        elif self.method in RERANK_METHODS:
            scores = self.__search_rrf_rerank(question)
        else:
            raise ValueError(f"Search is not implemented for method {self.method}")

        top_indexes = scores.argsort()[::-1][:top_k]

        return self.__build_result(question, top_indexes, scores)

    def __search_tfidf(self, question: str):
        """Score all chunks against a question with cosine similarity."""
        if self.chunk_matrix is None:
            raise ValueError("TF-IDF retriever must be fit before search")

        question_vector = self.vectorizer.transform([" ".join(self.__analyze(question))])
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

        query_embedding = self.embedder.transform([question], input_type="query")[0]
        return self.embedding_matrix @ query_embedding

    def __search_rrf(self, question: str, rank_constant: int = 60):
        """Combine BM25 and embedding ranks with reciprocal rank fusion."""
        bm25_scores = self.__search_bm25(question)
        embedding_scores = self.__search_embeddings(question)
        return self.__rrf_scores(
            [bm25_scores, embedding_scores],
            rank_constant,
            candidate_k=self.rrf_candidate_k,
        )

    def __search_rrf_rerank(self, question: str):
        """Rerank the strongest RRF candidates with a cross-encoder."""
        if self.reranker is None:
            raise ValueError("Reranker retriever must be fit before search")

        rrf_scores = self.__search_rrf(question)
        candidate_indexes = [
            index
            for index in np.argsort(rrf_scores)[::-1][:self.rrf_candidate_k]
            if rrf_scores[index] > 0
        ]
        reranked_scores = np.zeros(len(self.chunks), dtype=float)
        if not candidate_indexes:
            return reranked_scores

        candidate_texts = [self.embedding_texts[index] for index in candidate_indexes]
        candidate_scores = self.reranker.score(question, candidate_texts)
        for index, score in zip(candidate_indexes, candidate_scores):
            reranked_scores[index] = float(score)

        return reranked_scores

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
        """Build a stable fingerprint for embedding cache invalidation."""
        digest = hashlib.sha256()
        digest.update(b"raw-embedding-input-v1")
        digest.update(b"\0")
        for stop_word in sorted(self.stop_words):
            digest.update(stop_word.encode("utf-8"))
            digest.update(b"\0")
        for chunk in chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._searchable_text(chunk).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    def __tokenize_chunks(self, chunks: list[Chunk], texts: list[str]) -> list[list[str]]:
        """Load cached chunk lemmas where possible and analyze stale chunks."""
        return [
            self.__filter_tokens(self.__chunk_lemmas(chunk, text))
            for chunk, text in zip(chunks, texts)
        ]

    def __chunk_lemmas(self, chunk: Chunk, searchable_text: str) -> list[str]:
        """Return cached lemmas for one chunk, falling back to live lemmatization."""
        cached_chunk = self.lemma_cache.get(chunk.chunk_id)
        if (
            isinstance(cached_chunk, dict)
            and cached_chunk.get("text_hash") == self.__text_hash(searchable_text)
            and isinstance(cached_chunk.get("lemmas"), list)
        ):
            return [
                lemma
                for lemma in cached_chunk["lemmas"]
                if isinstance(lemma, str)
            ]

        return self.ice_tokenizer.lemmatIce(searchable_text)

    def __rrf_scores(self, ranked_score_lists, rank_constant: int, *, candidate_k: int):
        """Fuse only top candidates so low-ranked noise does not dominate."""
        fused = np.zeros(len(self.chunks), dtype=float)
        for scores in ranked_score_lists:
            ranked_indexes = np.argsort(scores)[::-1][:candidate_k]
            for rank, index in enumerate(ranked_indexes, start=1):
                if scores[index] <= 0:
                    continue
                fused[index] += 1.0 / (rank_constant + rank)
        return fused

    

    def _searchable_text(self, chunk: Chunk) -> str:
        """Weight titles/sections a little more than body text."""
        return "\n".join(
            [
                chunk.title,
                chunk.section,
                chunk.text,
            ]
        )

    def __analyze(self, text: str) -> list[str]:
        """Analyze text into lowercase Icelandic tokens for retrieval."""
        return self.__filter_tokens(self.ice_tokenizer.lemmatIce(text))

    def __filter_tokens(self, tokens: list[str]) -> list[str]:
        """Filter analyzed tokens for retrieval."""
        return [
            token
            for token in tokens
            if (
                len(token) > 1
                and token.casefold() not in self.stop_words
                and re.search(r"\w", token)
            )
        ]

    def __load_lemma_cache(self, path: Path | None) -> dict[str, Any]:
        """Load precomputed chunk lemmas if the cache file is available."""
        if path is None or not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            cache = json.load(file)

        chunks = cache.get("chunks", {})
        return chunks if isinstance(chunks, dict) else {}

    def __load_stop_words(self, path: Path | None) -> set[str]:
        """Load stop words and their lemmas for analyzer filtering."""
        if path is None:
            return set()

        if not path.exists():
            return set()

        words = {
            line.strip().casefold()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not words:
            return set()

        lemmas = {
            token.casefold()
            for token in self.ice_tokenizer.lemmatIce(" ".join(sorted(words)))
            if token and re.search(r"\w", token)
        }
        return words | lemmas

    def __text_hash(self, text: str) -> str:
        """Return the hash format used by the chunk lemma cache."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    


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
    retriever = Retriever(
        method,
        cache_dir=chunks_path.parent / "embedding_cache",
        stop_words_path=DEFAULT_STOP_WORDS_PATH,
        lemma_cache_path=chunks_path.parent / "chunk_lemmas.json",
    )
    retriever.fit(chunks)
    return retriever


if __name__ == "__main__":
    question = "Hvað get ég gert ef vara er gölluð?"
    for method in ["tfidf", "bm25"]:
        retriever = build_retriever(method)
        result = retriever.search(question)
        print(json.dumps(result, ensure_ascii=False, indent=2))
