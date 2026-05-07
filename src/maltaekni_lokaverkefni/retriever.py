"""TF-IDF retriever over processed retrieval chunks."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .types_classes import Chunk
    from .ice_tokenizer import IceTokenizer
except ImportError:  # Allows direct script execution during early experiments.
    from types_classes import Chunk
    from ice_tokenizer import IceTokenizer


class Retriever:
    """
    Lexical retriever for the MVP RAG pipeline.

    Explanation:
        Retriever indexes processed Chunk objects and returns the most relevant
        chunks for a user question. It supports TF-IDF and BM25 retrieval. Both
        methods use IceTokenizer for Icelandic tokenization and return results
        in the retrieval contract shape expected by the answer-generation code.

    Attributes:
        chunks: Chunk objects currently indexed by the retriever.
        method: Retrieval method name, currently "tfidf" or "bm25".
        ice_tokenizer: Icelandic tokenizer used by both retrieval methods.
        tokenized_chunks: Tokenized chunk texts used by BM25.
        bm25: BM25Okapi index when method is "bm25".
        vectorizer: TfidfVectorizer when method is "tfidf".
        chunk_matrix: TF-IDF matrix when method is "tfidf".

    Public methods:
        fit(chunks): Build the retrieval index from Chunk objects.
        search(question, top_k=3): Return ranked chunks for a question.
    """

    def __init__(self, method):
        """Initialize an unfitted retriever for a supported retrieval method."""
        self.chunks: list[Chunk] = []
        self.method = method
        self.ice_tokenizer = IceTokenizer()
        self.tokenized_chunks: list[list[str]] = []
        self.bm25 = None
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                analyzer=self.__analyze,
                lowercase=False,
            )
            self.chunk_matrix = None
        if method == 'bm25':
            self.vectorizer = None
            self.chunk_matrix = None
        if method == 'embeddings':
            raise Exception("Not available yet")
        if method == 'word2vec':
            raise Exception("Not available yet")
        if method not in {'tfidf', 'bm25', 'embeddings', 'word2vec'}:
            raise ValueError(f"Unknown retrieval method: {method}")

    def fit(self, chunks: list[Chunk]):
        """Fit the selected retrieval method on chunk text."""
        if not chunks:
            raise ValueError("Cannot fit retriever with no chunks")
        self.chunks = chunks
        texts = [self._searchable_text(chunk) for chunk in chunks]
        if self.method == 'tfidf':
            self.__fit_tfidf(texts)
        elif self.method == 'bm25':
            self.__fit_bm25(texts)
        
    def __fit_tfidf(self, texts):
        """Fit the TF-IDF vectorizer on searchable chunk texts."""
        self.chunk_matrix = self.vectorizer.fit_transform(texts)

    def __fit_bm25(self, texts):
        """Fit the BM25 index on tokenized searchable chunk texts."""
        self.tokenized_chunks = [self.__analyze(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        

    def search(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Return top chunks in the retrieval contract shape."""
        if self.method == 'tfidf':
            scores = self.__search_tfidf(question)
        elif self.method == 'bm25':
            scores = self.__search_bm25(question)
        else:
            raise ValueError(f"Search is not implemented for method {self.method}")

        top_indexes = scores.argsort()[::-1][:top_k]

        return self.__build_result(question, top_indexes, scores)

    def __search_tfidf(self, question: str):
        """Score all chunks against a question with cosine similarity."""
        if self.chunk_matrix is None:
            raise ValueError("TF-IDF retriever must be fit before search")

        question_vector = self.vectorizer.transform([self.__expand_question(question)])
        scores = cosine_similarity(question_vector, self.chunk_matrix).flatten()
        return scores

    def __search_bm25(self, question: str):
        """Score all chunks against a question with BM25."""
        if self.bm25 is None:
            raise ValueError("BM25 retriever must be fit before search")

        return self.bm25.get_scores(self.__analyze(self.__expand_question(question)))

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
        try:
            tokens = self.ice_tokenizer.tokenIce(text)
        except Exception:
            tokens = re.findall(r"[\wáðéíóúýþæö]+", text.lower())

        return [
            token
            for token in tokens
            if len(token) > 1 and re.search(r"[\wáðéíóúýþæö]", token)
        ]

    def __expand_question(self, question: str) -> str:
        """Add a few transparent consumer-rights synonyms before retrieval."""
        lower_question = question.lower()
        additions = []

        if any(word in lower_question for word in ["vara", "vöru", "vörur", "hlutur"]):
            additions.extend(["söluhlutur", "hlutur", "vara"])

        if any(word in lower_question for word in ["gölluð", "gallað", "gallaður", "galli", "galla"]):
            additions.extend(["galli", "galla", "gallaður", "úrræði", "neytanda"])

        if any(word in lower_question for word in ["netkaup", "fjarsala", "skila", "skilaréttur"]):
            additions.extend(["fjarsölusamningur", "falla frá samningi", "uppsögn", "skilaréttur"])

        if any(word in lower_question for word in ["kvarta", "kvörtun", "tilkynna"]):
            additions.extend(["tilkynning", "kvörtun", "galla", "seljanda"])

        return " ".join([question, *additions])


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
    retriever = Retriever(method)
    retriever.fit(chunks)
    return retriever


if __name__ == "__main__":
    question = "Hvað get ég gert ef vara er gölluð?"
    for method in ["tfidf", "bm25"]:
        retriever = build_retriever(method)
        result = retriever.search(question)
        print(json.dumps(result, ensure_ascii=False, indent=2))
