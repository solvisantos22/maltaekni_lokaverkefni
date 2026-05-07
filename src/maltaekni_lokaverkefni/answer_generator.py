"""Build grounded answers from retrieved consumer-rights source chunks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

try:
    from .prompts import SYSTEM_PROMPT, build_answer_prompt
except ImportError:  # Allows direct script execution during early experiments.
    from prompts import SYSTEM_PROMPT, build_answer_prompt


UNCERTAIN_ANSWER = (
    "Ég finn ekki nægar upplýsingar í heimildunum til að svara þessu örugglega."
)


@dataclass(frozen=True)
class SourceReference:
    """One cited source chunk shown with the generated answer."""

    citation_id: int
    chunk_id: str
    title: str
    source: str
    section: str
    url: str
    text: str
    score: float | None = None
    retrieval_method: str | None = None


@dataclass(frozen=True)
class AnswerResult:
    """Grounded answer plus sources and prompt context."""

    question: str
    answer: str
    sources: list[SourceReference]
    system_prompt: str
    user_prompt: str
    confidence: str
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.__dict__ for source in self.sources],
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "confidence": self.confidence,
            "method": self.method,
        }


def generate_grounded_answer(
    retrieval_result: dict[str, Any],
    max_sources: int = 3,
    method: str = "extractive_fallback",
) -> AnswerResult:
    """Generate a citation-grounded answer from retrieval contract output."""
    question = retrieval_result.get("question", "")
    chunks = retrieval_result.get("chunks", [])[:max_sources]
    user_prompt = build_answer_prompt(question, chunks, max_chunks=max_sources)
    sources = [
        _source_from_chunk(index=index, chunk=chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]

    if not sources:
        return AnswerResult(
            question=question,
            answer=UNCERTAIN_ANSWER,
            sources=[],
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            confidence="low",
            method=method,
        )

    answer = _build_extractive_answer(question, sources)
    confidence = _estimate_confidence(sources)

    return AnswerResult(
        question=question,
        answer=answer,
        sources=sources,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        confidence=confidence,
        method=method,
    )


def _source_from_chunk(index: int, chunk: dict[str, Any]) -> SourceReference:
    return SourceReference(
        citation_id=index,
        chunk_id=str(chunk.get("chunk_id", "")),
        title=str(chunk.get("title", "Óþekktur titill")),
        source=str(chunk.get("source", "Óþekkt heimild")),
        section=str(chunk.get("section", "Ótilgreint")),
        url=str(chunk.get("url", "")),
        text=str(chunk.get("text", "")),
        score=_optional_float(chunk.get("score")),
        retrieval_method=chunk.get("retrieval_method"),
    )


def _build_extractive_answer(question: str, sources: list[SourceReference]) -> str:
    question_terms = _content_terms(question)
    cited_sentences = []

    for source in sources:
        remedies = _extract_remedy_list(source.text)
        if remedies:
            cited_sentences.append((remedies, source.citation_id))
            break

        sentence = _best_sentence(source.text, question_terms)
        if sentence:
            cited_sentences.append((sentence, source.citation_id))

    if not cited_sentences:
        return UNCERTAIN_ANSWER

    lead = "Samkvæmt heimildunum má svara þessu svona:"
    sentences = [
        f"{sentence} [{citation_id}]"
        for sentence, citation_id in cited_sentences[:2]
    ]
    source_ids = ", ".join(f"[{source.citation_id}]" for source in sources[: len(sentences)])

    return f"{lead} {' '.join(sentences)} Heimildir: {source_ids}"


def _extract_remedy_list(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not any("getur neytandi:" in line.lower() for line in lines):
        return ""

    remedies = []
    for line in lines:
        match = re.match(r"^[a-e]\.\s+(.+?);?$", line)
        if match:
            remedies.append(match.group(1).rstrip(";"))

    if not remedies:
        return ""

    remedy_text = ", ".join(remedies)
    ending = "" if remedy_text.endswith(".") else "."
    return f"Ef söluhlutur reynist gallaður getur neytandi meðal annars {remedy_text}{ending}"


def _best_sentence(text: str, question_terms: set[str]) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    def score(sentence: str) -> tuple[int, int]:
        sentence_terms = _content_terms(sentence)
        return (len(question_terms & sentence_terms), len(sentence_terms))

    best = max(sentences, key=score)
    return best if score(best)[0] > 0 else sentences[0]


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    sentence_candidates = re.split(r"(?<=[.!?])\s+", normalized)
    return [
        sentence.strip()
        for sentence in sentence_candidates
        if 30 <= len(sentence.strip()) <= 450
    ]


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "að",
        "af",
        "á",
        "ég",
        "ef",
        "ekki",
        "en",
        "er",
        "fyrir",
        "get",
        "hvað",
        "í",
        "með",
        "og",
        "sem",
        "til",
        "um",
        "við",
        "það",
    }
    return {
        token
        for token in re.findall(r"[\wáðéíóúýþæö]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _estimate_confidence(sources: list[SourceReference]) -> str:
    best_score = max((source.score or 0.0) for source in sources)
    if best_score >= 10 or best_score >= 0.2:
        return "medium"
    return "low"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
