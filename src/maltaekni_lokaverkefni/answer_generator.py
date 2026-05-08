"""Build grounded answers from retrieved consumer-rights source chunks."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

import httpx

try:
    from .prompts import SYSTEM_PROMPT, build_answer_prompt
except ImportError:  # Allows direct script execution during early experiments.
    from prompts import SYSTEM_PROMPT, build_answer_prompt

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is helpful locally, but environment variables are enough.
    load_dotenv = None


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

    llm_answer, llm_method = _build_llm_answer(SYSTEM_PROMPT, user_prompt)
    if llm_answer:
        answer = _ensure_source_line(llm_answer, sources)
        method = llm_method
    else:
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


def _build_llm_answer(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """Call the configured LLM provider for a grounded Icelandic answer."""
    if load_dotenv is not None:
        load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if provider in {"none", "off", "disabled"}:
        return "", ""

    if provider not in {"auto", "gemini", "openai"}:
        provider = "auto"

    if provider in {"auto", "gemini"}:
        answer = _build_gemini_answer(system_prompt, user_prompt)
        if _is_incomplete_llm_answer(answer):
            answer = _build_gemini_answer(
                system_prompt,
                user_prompt,
                max_output_tokens=max(4096, _llm_max_output_tokens()),
            )
        if answer:
            return answer, f"gemini:{_gemini_model()}"

    if provider in {"auto", "openai"}:
        answer = _build_openai_answer(system_prompt, user_prompt)
        if answer:
            return answer, f"openai:{_openai_model()}"

    return "", ""


def _build_gemini_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    max_output_tokens: int | None = None,
) -> str:
    """Call Gemini to turn retrieved chunks into a grounded Icelandic answer."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            },
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens or _llm_max_output_tokens(),
            "thinkingConfig": {
                "thinkingLevel": os.getenv("GEMINI_THINKING_LEVEL", "low"),
            },
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    timeout = _llm_timeout_seconds()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_gemini_model()}:generateContent"
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
    except (httpx.HTTPError, ValueError):
        return ""

    return _extract_gemini_text(response_data).strip()


def _build_openai_answer(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI to turn retrieved chunks into a grounded Icelandic answer."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    payload = {
        "model": _openai_model(),
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": _llm_max_output_tokens(),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = _llm_timeout_seconds()

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            response_data = response.json()
    except (httpx.HTTPError, ValueError):
        return ""

    return _extract_openai_text(response_data).strip()


def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


def _openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _llm_max_output_tokens() -> int:
    return _positive_int_from_env("LLM_MAX_OUTPUT_TOKENS", default=4096)


def _llm_timeout_seconds() -> float:
    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0

    return timeout if timeout > 0 else 30.0


def _positive_int_from_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default

    return value if value > 0 else default


def _extract_gemini_text(response_data: dict[str, Any]) -> str:
    """Extract visible text from a Gemini generateContent JSON payload."""
    text_parts: list[str] = []
    for candidate in response_data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if part.get("text"):
                text_parts.append(str(part["text"]))

    return "\n".join(text_parts)


def _is_incomplete_llm_answer(answer: str) -> bool:
    """Detect obviously truncated model text before it is shown to the user."""
    stripped = answer.strip()
    if not stripped:
        return True

    first_character = stripped[0]
    starts_like_sentence = first_character.isupper() or first_character.isdigit()
    has_source_line = bool(re.search(r"Heimildir:\s*\[\d+\]", stripped))
    has_minimum_length = len(stripped) >= 120

    return not (starts_like_sentence and has_source_line and has_minimum_length)


def _extract_openai_text(response_data: dict[str, Any]) -> str:
    """Extract visible text from a Responses API JSON payload."""
    if isinstance(response_data.get("output_text"), str):
        return response_data["output_text"]

    text_parts: list[str] = []
    for item in response_data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_parts.append(str(content["text"]))

    return "\n".join(text_parts)


def _ensure_source_line(answer: str, sources: list[SourceReference]) -> str:
    """Keep the UI citation contract even if the model omits the final source line."""
    if re.search(r"Heimildir:\s*\[\d+\]", answer):
        return answer

    cited_ids = sorted(
        {
            int(match)
            for match in re.findall(r"\[(\d+)\]", answer)
            if int(match) <= len(sources)
        }
    )
    if not cited_ids:
        cited_ids = [sources[0].citation_id]

    source_line = ", ".join(f"[{citation_id}]" for citation_id in cited_ids)
    return f"{answer.rstrip()}\n\nHeimildir: {source_line}"


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
