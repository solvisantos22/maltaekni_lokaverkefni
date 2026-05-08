"""Run report-ready evaluation over retrieval and answer-generation methods."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from .answer_generator import generate_grounded_answer
    from .retriever import build_retriever
except ImportError:  # Allows direct script execution during early experiments.
    from answer_generator import generate_grounded_answer
    from retriever import build_retriever


DEFAULT_METHODS = ["tfidf", "bm25"]
DEFAULT_QUESTIONS_PATH = Path("docs/evaluation_questions.csv")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.json")
DEFAULT_OUTPUT_DIR = Path("reports/evaluation")


@dataclass(frozen=True)
class EvaluationRow:
    """One question/method evaluation result."""

    question_id: str
    question: str
    topic: str
    expected_source_hint: str
    retrieval_method: str
    answer_method: str
    top_1_chunk_id: str
    top_1_title: str
    top_1_section: str
    top_1_score: float | None
    top_3_sections: str
    expected_hint_in_top_3: bool
    confidence: str
    latency_seconds: float
    answer: str
    source_urls: str
    error: str = ""


def main() -> None:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv()

    if args.no_llm:
        disable_llm_calls()

    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    retrievers = {
        method: build_retriever(method, chunks_path=args.chunks)
        for method in args.methods
    }

    rows: list[EvaluationRow] = []
    details: list[dict[str, Any]] = []

    for question in questions:
        for method, retriever in retrievers.items():
            started = perf_counter()
            try:
                retrieval_result = retriever.search(question["question"], top_k=args.top_k)
                answer_result = generate_grounded_answer(
                    retrieval_result,
                    max_sources=args.top_k,
                )
                latency = perf_counter() - started
                rows.append(
                    build_row(
                        question=question,
                        retrieval_method=method,
                        answer_result=answer_result.to_dict(),
                        latency_seconds=latency,
                    )
                )
                details.append(
                    {
                        "question": question,
                        "retrieval_method": method,
                        "latency_seconds": latency,
                        "retrieval_result": retrieval_result,
                        "answer_result": answer_result.to_dict(),
                    }
                )
            except Exception as error:  # Keep long batch runs useful after one failure.
                latency = perf_counter() - started
                rows.append(
                    EvaluationRow(
                        question_id=question["id"],
                        question=question["question"],
                        topic=question.get("topic", ""),
                        expected_source_hint=question.get("expected_source_hint", ""),
                        retrieval_method=method,
                        answer_method="error",
                        top_1_chunk_id="",
                        top_1_title="",
                        top_1_section="",
                        top_1_score=None,
                        top_3_sections="",
                        expected_hint_in_top_3=False,
                        confidence="",
                        latency_seconds=latency,
                        answer="",
                        source_urls="",
                        error=f"{type(error).__name__}: {error}",
                    )
                )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    summary_path = output_dir / f"evaluation_summary_{stamp}.csv"
    details_path = output_dir / f"evaluation_details_{stamp}.jsonl"
    latest_summary_path = output_dir / "evaluation_summary_latest.csv"
    latest_details_path = output_dir / "evaluation_details_latest.jsonl"

    write_summary(summary_path, rows)
    write_summary(latest_summary_path, rows)
    write_details(details_path, details)
    write_details(latest_details_path, details)

    print(f"Wrote {summary_path}")
    print(f"Wrote {details_path}")
    print_overview(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval methods and grounded answers on CSV questions.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Retrieval methods, e.g. tfidf bm25 rrf-bge-m3-bm25.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions. Useful for quick checks.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable external LLM calls and use the local extractive fallback.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def load_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row.get("id") and row.get("question")
        ]


def disable_llm_calls() -> None:
    os.environ["LLM_PROVIDER"] = "none"
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)


def build_row(
    *,
    question: dict[str, str],
    retrieval_method: str,
    answer_result: dict[str, Any],
    latency_seconds: float,
) -> EvaluationRow:
    sources = answer_result.get("sources", [])
    top_source = sources[0] if sources else {}
    top_3_sections = " | ".join(source.get("section", "") for source in sources[:3])
    expected_hint = question.get("expected_source_hint", "")

    return EvaluationRow(
        question_id=question["id"],
        question=question["question"],
        topic=question.get("topic", ""),
        expected_source_hint=expected_hint,
        retrieval_method=retrieval_method,
        answer_method=answer_result.get("method", ""),
        top_1_chunk_id=top_source.get("chunk_id", ""),
        top_1_title=top_source.get("title", ""),
        top_1_section=top_source.get("section", ""),
        top_1_score=top_source.get("score"),
        top_3_sections=top_3_sections,
        expected_hint_in_top_3=hint_matches(expected_hint, top_3_sections),
        confidence=answer_result.get("confidence", ""),
        latency_seconds=round(latency_seconds, 3),
        answer=answer_result.get("answer", ""),
        source_urls=" | ".join(source.get("url", "") for source in sources[:3]),
    )


def hint_matches(expected_hint: str, retrieved_sections: str) -> bool:
    if not expected_hint:
        return False

    return normalize_text(expected_hint) in normalize_text(retrieved_sections)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def write_summary(path: Path, rows: list[EvaluationRow]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_details(path: Path, details: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in details:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_overview(rows: list[EvaluationRow]) -> None:
    methods = sorted({row.retrieval_method for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row.retrieval_method == method]
        total = len(method_rows)
        errors = sum(1 for row in method_rows if row.error)
        hint_hits = sum(1 for row in method_rows if row.expected_hint_in_top_3)
        avg_latency = sum(row.latency_seconds for row in method_rows) / max(total, 1)
        print(
            f"{method}: {hint_hits}/{total} hint hits in top 3, "
            f"{errors} errors, avg {avg_latency:.2f}s"
        )


if __name__ == "__main__":
    main()
