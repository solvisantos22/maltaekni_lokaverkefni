"""Build grounded answers from retrieved consumer-rights source chunks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any

import httpx

try:
    from .prompts import build_answer_prompt, get_prompt_profile, get_system_prompt
except ImportError:  # Allows direct script execution during early experiments.
    from prompts import build_answer_prompt, get_prompt_profile, get_system_prompt

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is helpful locally, but environment variables are enough.
    load_dotenv = None


UNCERTAIN_ANSWER = (
    "Ég finn ekki nægar upplýsingar í heimildunum til að svara þessu örugglega."
)
STOP_WORDS_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "all_stop_words.txt"


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
    reason: str = ""


@dataclass(frozen=True)
class AnswerResult:
    """Grounded answer plus sources and prompt context."""

    question: str
    answer: str
    sources: list[SourceReference]
    system_prompt: str
    user_prompt: str
    prompt_profile: str
    confidence: str
    confidence_reason: str
    method: str
    usage: dict[str, Any]
    source_coverage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the answer result into the API/evaluation response shape."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.__dict__ for source in self.sources],
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "prompt_profile": self.prompt_profile,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "method": self.method,
            "usage": self.usage,
            "source_coverage": self.source_coverage,
        }


def generate_grounded_answer(
    retrieval_result: dict[str, Any],
    max_sources: int = 3,
    method: str = "extractive_fallback",
) -> AnswerResult:
    """Generate a citation-grounded answer from retrieval contract output."""
    question = retrieval_result.get("question", "")
    chunks = retrieval_result.get("chunks", [])[:max_sources]
    prompt_profile = get_prompt_profile()
    system_prompt = get_system_prompt(prompt_profile)
    user_prompt = build_answer_prompt(question, chunks, max_chunks=max_sources)
    question_terms = _content_terms(question)
    sources = [
        _source_from_chunk(index=index, chunk=chunk, question_terms=question_terms)
        for index, chunk in enumerate(chunks, start=1)
    ]

    if not sources:
        return AnswerResult(
            question=question,
            answer=UNCERTAIN_ANSWER,
            sources=[],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_profile=prompt_profile,
            confidence="low",
            confidence_reason="Engar heimildir fundust fyrir spurninguna.",
            method=method,
            usage={},
            source_coverage={
                "cited_source_count": 0,
                "source_count": 0,
                "coverage_ratio": 0,
                "cited_source_ids": [],
                "uncited_source_ids": [],
            },
        )

    llm_answer, llm_method, usage = _build_llm_answer(system_prompt, user_prompt)
    if llm_answer:
        answer = _ensure_source_line(llm_answer, sources)
        method = llm_method
    else:
        answer = _build_extractive_answer(question, sources)
        usage = {}

    confidence, confidence_reason = _estimate_confidence(
        sources=sources,
        answer=answer,
        method=method,
    )
    source_coverage = _source_coverage(answer, sources)

    return AnswerResult(
        question=question,
        answer=answer,
        sources=sources,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_profile=prompt_profile,
        confidence=confidence,
        confidence_reason=confidence_reason,
        method=method,
        usage=usage,
        source_coverage=source_coverage,
    )


def _build_llm_answer(system_prompt: str, user_prompt: str) -> tuple[str, str, dict[str, Any]]:
    """Call the configured LLM provider for a grounded Icelandic answer."""
    if load_dotenv is not None:
        load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if provider in {"none", "off", "disabled"}:
        return "", "", {}

    if provider not in {"auto", "gemini", "openai"}:
        provider = "auto"

    if provider in {"auto", "gemini"}:
        answer, usage = _build_gemini_answer(system_prompt, user_prompt)
        if _is_incomplete_llm_answer(answer):
            answer, usage = _build_gemini_answer(
                system_prompt,
                user_prompt,
                max_output_tokens=max(4096, _llm_max_output_tokens()),
            )
        if answer:
            return answer, f"gemini:{_gemini_model()}", usage

    if provider in {"auto", "openai"}:
        answer, usage = _build_openai_answer(system_prompt, user_prompt)
        if answer:
            return answer, f"openai:{_openai_model()}", usage

    return "", "", {}


def _build_gemini_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    max_output_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call Gemini to turn retrieved chunks into a grounded Icelandic answer."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "", {}

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
        return "", {}

    return _extract_gemini_text(response_data).strip(), _extract_gemini_usage(response_data)


def _build_openai_answer(system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
    """Call OpenAI to turn retrieved chunks into a grounded Icelandic answer."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "", {}

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
        return "", {}

    return _extract_openai_text(response_data).strip(), _extract_openai_usage(response_data)


def _gemini_model() -> str:
    """Return the configured Gemini model name or the project default."""
    return os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


def _openai_model() -> str:
    """Return the configured OpenAI model name or the fallback default."""
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _llm_max_output_tokens() -> int:
    """Return the maximum answer length used for provider calls."""
    return _positive_int_from_env("LLM_MAX_OUTPUT_TOKENS", default=4096)


def _llm_timeout_seconds() -> float:
    """Return a positive HTTP timeout for LLM calls."""
    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0

    return timeout if timeout > 0 else 30.0


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer environment setting with a safe default."""
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


def _extract_gemini_usage(response_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize Gemini usage metadata for evaluation reports."""
    usage_metadata = response_data.get("usageMetadata", {})
    prompt_tokens = _optional_int(usage_metadata.get("promptTokenCount"))
    output_tokens = _optional_int(usage_metadata.get("candidatesTokenCount"))
    thought_tokens = _optional_int(usage_metadata.get("thoughtsTokenCount"))
    total_tokens = _optional_int(usage_metadata.get("totalTokenCount"))
    return _usage_payload(
        provider="gemini",
        model=_gemini_model(),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        thought_tokens=thought_tokens,
        total_tokens=total_tokens,
    )


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


def _extract_openai_usage(response_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI Responses API usage metadata for evaluation reports."""
    usage = response_data.get("usage", {})
    prompt_tokens = _optional_int(usage.get("input_tokens"))
    output_tokens = _optional_int(usage.get("output_tokens"))
    total_tokens = _optional_int(usage.get("total_tokens"))
    return _usage_payload(
        provider="openai",
        model=_openai_model(),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        thought_tokens=None,
        total_tokens=total_tokens,
    )


def _usage_payload(
    *,
    provider: str,
    model: str,
    prompt_tokens: int | None,
    output_tokens: int | None,
    thought_tokens: int | None,
    total_tokens: int | None,
) -> dict[str, Any]:
    """Build the shared usage dictionary stored by API and evaluation runs."""
    return {
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": _estimate_cost_usd(
            provider=provider,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
        ),
    }


def _estimate_cost_usd(
    *,
    provider: str,
    prompt_tokens: int | None,
    output_tokens: int | None,
    thought_tokens: int | None,
) -> float | None:
    """Estimate cost only when local per-million-token rates are configured."""
    prefix = provider.upper()
    input_rate = _optional_float(os.getenv(f"{prefix}_INPUT_COST_PER_1M"))
    output_rate = _optional_float(os.getenv(f"{prefix}_OUTPUT_COST_PER_1M"))
    if input_rate is None and output_rate is None:
        return None

    input_cost = ((prompt_tokens or 0) / 1_000_000) * (input_rate or 0)
    output_billable_tokens = (output_tokens or 0) + (thought_tokens or 0)
    output_cost = (output_billable_tokens / 1_000_000) * (output_rate or 0)
    return round(input_cost + output_cost, 8)


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


def _source_coverage(answer: str, sources: list[SourceReference]) -> dict[str, Any]:
    """Measure how many retrieved sources are actually cited in the answer."""
    source_ids = {source.citation_id for source in sources}
    cited_ids = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", answer)
        if int(match) in source_ids
    }
    uncited_ids = source_ids - cited_ids
    source_count = len(source_ids)
    return {
        "cited_source_count": len(cited_ids),
        "source_count": source_count,
        "coverage_ratio": round(len(cited_ids) / source_count, 4) if source_count else 0,
        "cited_source_ids": sorted(cited_ids),
        "uncited_source_ids": sorted(uncited_ids),
    }


def _source_from_chunk(
    index: int,
    chunk: dict[str, Any],
    question_terms: set[str],
) -> SourceReference:
    """Convert one retriever chunk dictionary into a cited source object."""
    text = str(chunk.get("text", ""))
    title = str(chunk.get("title", "Óþekktur titill"))
    section = str(chunk.get("section", "Ótilgreint"))
    return SourceReference(
        citation_id=index,
        chunk_id=str(chunk.get("chunk_id", "")),
        title=title,
        source=str(chunk.get("source", "Óþekkt heimild")),
        section=section,
        url=str(chunk.get("url", "")),
        text=text,
        score=_optional_float(chunk.get("score")),
        retrieval_method=chunk.get("retrieval_method"),
        reason=_source_reason(
            citation_id=index,
            title=title,
            section=section,
            text=text,
            score=_optional_float(chunk.get("score")),
            question_terms=question_terms,
        ),
    )


def _source_reason(
    *,
    citation_id: int,
    title: str,
    section: str,
    text: str,
    score: float | None,
    question_terms: set[str],
) -> str:
    """Explain source selection in simple reportable language."""
    searchable = _content_terms(" ".join([title, section, text[:700]]))
    overlap = _matching_terms(question_terms, searchable)
    score_text = f" Vægi heimildar er {score:.4g}." if score is not None else ""

    if overlap:
        terms = ", ".join(overlap[:4])
        return (
            f"Heimild [{citation_id}] var valin vegna þess að kaflinn tengist "
            f"spurningunni með lykilorðum eins og {terms}.{score_text}"
        )

    return (
        f"Heimild [{citation_id}] var valin af leitarkerfinu sem eitt af efstu "
        f"textabrotunum fyrir spurninguna.{score_text}"
    )


def _matching_terms(question_terms: set[str], source_terms: set[str]) -> list[str]:
    """Find direct and light-normalized term overlap for Icelandic source reasons."""
    direct_overlap = question_terms & source_terms
    if direct_overlap:
        return sorted(direct_overlap)

    source_keys = {_term_key(term) for term in source_terms}
    return sorted(
        term
        for term in question_terms
        if _term_key(term) in source_keys
    )


def _term_key(term: str) -> str:
    """Build a short normalized key for lightweight Icelandic term matching."""
    normalized = term.translate(str.maketrans({"ö": "a", "ó": "o", "á": "a"}))
    return normalized[:4]


def _build_extractive_answer(question: str, sources: list[SourceReference]) -> str:
    """Build a deterministic fallback answer when no LLM provider is available."""
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
    """Extract enumerated consumer remedies from article text when present."""
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
    """Choose the sentence with the strongest direct overlap with the question."""
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    def score(sentence: str) -> tuple[int, int]:
        """Prefer overlap first, then more informative longer sentences."""
        sentence_terms = _content_terms(sentence)
        return (len(question_terms & sentence_terms), len(sentence_terms))

    best = max(sentences, key=score)
    return best if score(best)[0] > 0 else sentences[0]


def _split_sentences(text: str) -> list[str]:
    """Split source text into sentence-sized snippets suitable for fallback answers."""
    normalized = " ".join(text.split())
    sentence_candidates = re.split(r"(?<=[.!?])\s+", normalized)
    return [
        sentence.strip()
        for sentence in sentence_candidates
        if 30 <= len(sentence.strip()) <= 450
    ]


def _content_terms(text: str) -> set[str]:
    """Return lowercase non-stopword terms for simple overlap heuristics."""
    stopwords = _load_stop_words()
    return {
        token
        for token in re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
        if len(token) > 2 and token not in stopwords
    }


@lru_cache(maxsize=1)
def _load_stop_words() -> set[str]:
    """Load shared Icelandic stopwords used by lightweight answer heuristics."""
    if not STOP_WORDS_PATH.exists():
        return set()

    return {
        line.strip().casefold()
        for line in STOP_WORDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _estimate_confidence(
    *,
    sources: list[SourceReference],
    answer: str,
    method: str,
) -> tuple[str, str]:
    """Assign a coarse confidence label from citations and retrieval scores."""
    if not sources:
        return "low", "Engar heimildir fundust."

    if UNCERTAIN_ANSWER in answer:
        return "low", "Svarið segir sjálft að heimildirnar nægi ekki."

    best_score = max((source.score or 0.0) for source in sources)
    cited_ids = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", answer)
        if int(match) <= len(sources)
    }
    has_source_line = bool(re.search(r"Heimildir:\s*\[\d+\]", answer))
    strong_retrieval = best_score >= 10 or best_score >= 0.2
    enough_citations = len(cited_ids) >= min(2, len(sources))
    llm_answer = method.startswith(("gemini:", "openai:"))

    if llm_answer and strong_retrieval and enough_citations and has_source_line:
        return (
            "high",
            "Sterkustu heimildirnar hafa hátt vægi og svarið vísar í fleiri en eina heimild.",
        )

    if strong_retrieval and cited_ids:
        return (
            "medium",
            "Að minnsta kosti ein heimild hefur gott vægi og svarið inniheldur tilvísanir.",
        )

    return (
        "low",
        "Heimildir eða tilvísanir eru veikar og því ætti að lesa svarið með varúð.",
    )


def _optional_float(value: Any) -> float | None:
    """Parse a float-like value, returning None for missing or invalid input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    """Parse an integer-like value, returning None for missing or invalid input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
