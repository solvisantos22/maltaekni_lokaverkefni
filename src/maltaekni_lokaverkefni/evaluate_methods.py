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
from types_classes import EvaluationRow

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



def main() -> None:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv()

    apply_experiment_overrides(args)
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
                        run_label=args.run_label,
                    )
                )
                details.append(
                    {
                        "run_label": args.run_label,
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
                        run_label=args.run_label,
                        question_id=question["id"],
                        question=question["question"],
                        topic=question.get("topic", ""),
                        case_type=question.get("case_type", ""),
                        expected_behavior=question.get("expected_behavior", ""),
                        expected_relevant_section=expected_relevant_section(question),
                        retrieval_method=method,
                        answer_method="error",
                        top_1_chunk_id="",
                        top_1_title="",
                        top_1_section="",
                        top_1_score=None,
                        top_3_sections="",
                        retrieval_check_applicable=retrieval_check_applicable(
                            expected_relevant_section(question)
                        ),
                        expected_section_in_top_3=False,
                        confidence="",
                        confidence_reason="",
                        prompt_profile=os.getenv("PROMPT_PROFILE", "balanced"),
                        cited_source_count=None,
                        source_count=None,
                        source_coverage_ratio=None,
                        llm_provider="",
                        llm_model="",
                        prompt_tokens=None,
                        output_tokens=None,
                        thought_tokens=None,
                        total_tokens=None,
                        estimated_cost_usd=None,
                        latency_seconds=latency,
                        answer="",
                        source_urls="",
                        error=f"{type(error).__name__}: {error}",
                    )
                )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    summary_path = output_dir / f"evaluation_summary_{stamp}.csv"
    details_path = output_dir / f"evaluation_details_{stamp}.jsonl"
    method_summary_path = output_dir / f"evaluation_method_summary_{stamp}.csv"
    latest_summary_path = output_dir / "evaluation_summary_latest.csv"
    latest_details_path = output_dir / "evaluation_details_latest.jsonl"
    latest_method_summary_path = output_dir / "evaluation_method_summary_latest.csv"

    write_summary(summary_path, rows)
    write_summary(latest_summary_path, rows)
    write_details(details_path, details)
    write_details(latest_details_path, details)
    method_summary = build_method_summary(rows)
    write_dict_rows(method_summary_path, method_summary)
    write_dict_rows(latest_method_summary_path, method_summary)

    print(f"Wrote {summary_path}")
    print(f"Wrote {details_path}")
    print(f"Wrote {method_summary_path}")
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
    parser.add_argument(
        "--run-label",
        default="default",
        help="Label for report-ready model/prompt comparison runs.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "gemini", "openai", "none"],
        default=None,
        help="Override LLM_PROVIDER for this evaluation run.",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Override GEMINI_MODEL for this evaluation run.",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help="Override OPENAI_MODEL for this evaluation run.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["balanced", "strict", "user_friendly"],
        default=None,
        help="Prompt variant to use for answer-generation comparisons.",
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


def apply_experiment_overrides(args: argparse.Namespace) -> None:
    """Apply run-scoped model and prompt settings without editing .env."""
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.gemini_model:
        os.environ["GEMINI_MODEL"] = args.gemini_model
    if args.openai_model:
        os.environ["OPENAI_MODEL"] = args.openai_model
    if args.prompt_profile:
        os.environ["PROMPT_PROFILE"] = args.prompt_profile


def build_row(
    *,
    question: dict[str, str],
    retrieval_method: str,
    answer_result: dict[str, Any],
    latency_seconds: float,
    run_label: str,
) -> EvaluationRow:
    sources = answer_result.get("sources", [])
    top_source = sources[0] if sources else {}
    top_3_sections = " | ".join(source.get("section", "") for source in sources[:3])
    expected_section = expected_relevant_section(question)
    usage = answer_result.get("usage", {})
    source_coverage = answer_result.get("source_coverage", {})
    check_applicable = retrieval_check_applicable(expected_section)

    return EvaluationRow(
        run_label=run_label,
        question_id=question["id"],
        question=question["question"],
        topic=question.get("topic", ""),
        case_type=question.get("case_type", ""),
        expected_behavior=question.get("expected_behavior", ""),
        expected_relevant_section=expected_section,
        retrieval_method=retrieval_method,
        answer_method=answer_result.get("method", ""),
        top_1_chunk_id=top_source.get("chunk_id", ""),
        top_1_title=top_source.get("title", ""),
        top_1_section=top_source.get("section", ""),
        top_1_score=top_source.get("score"),
        top_3_sections=top_3_sections,
        retrieval_check_applicable=check_applicable,
        expected_section_in_top_3=(
            section_matches(expected_section, top_3_sections) if check_applicable else False
        ),
        confidence=answer_result.get("confidence", ""),
        confidence_reason=answer_result.get("confidence_reason", ""),
        prompt_profile=answer_result.get("prompt_profile", ""),
        cited_source_count=source_coverage.get("cited_source_count"),
        source_count=source_coverage.get("source_count"),
        source_coverage_ratio=source_coverage.get("coverage_ratio"),
        llm_provider=usage.get("provider", ""),
        llm_model=usage.get("model", ""),
        prompt_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("output_tokens"),
        thought_tokens=usage.get("thought_tokens"),
        total_tokens=usage.get("total_tokens"),
        estimated_cost_usd=usage.get("estimated_cost_usd"),
        latency_seconds=round(latency_seconds, 3),
        answer=answer_result.get("answer", ""),
        source_urls=" | ".join(source.get("url", "") for source in sources[:3]),
    )


def section_matches(expected_section: str, retrieved_sections: str) -> bool:
    if not expected_section:
        return False

    return normalize_text(expected_section) in normalize_text(retrieved_sections)


def retrieval_check_applicable(expected_section: str) -> bool:
    normalized = normalize_text(expected_section)
    return bool(normalized) and normalized not in {"no_relevant_source", "none", "n/a"}


def expected_relevant_section(question: dict[str, str]) -> str:
    """Read the current column name while accepting older evaluation CSV files."""
    return question.get("expected_relevant_section") or question.get("expected_source_hint", "")


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def write_summary(path: Path, rows: list[EvaluationRow]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_details(path: Path, details: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in details:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_method_summary(rows: list[EvaluationRow]) -> list[dict[str, Any]]:
    """Aggregate report-ready automatic metrics by retrieval method."""
    summaries: list[dict[str, Any]] = []
    methods = sorted({row.retrieval_method for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row.retrieval_method == method]
        applicable_rows = [row for row in method_rows if row.retrieval_check_applicable]
        total = len(method_rows)
        errors = sum(1 for row in method_rows if row.error)
        section_hits = sum(1 for row in applicable_rows if row.expected_section_in_top_3)
        latencies = [row.latency_seconds for row in method_rows if row.latency_seconds is not None]
        summaries.append(
            {
                "retrieval_method": method,
                "run_label": method_rows[0].run_label if method_rows else "",
                "prompt_profile": method_rows[0].prompt_profile if method_rows else "",
                "questions": total,
                "retrieval_check_questions": len(applicable_rows),
                "errors": errors,
                "expected_section_top3_hits": section_hits,
                "expected_section_top3_rate": (
                    round(section_hits / len(applicable_rows), 4) if applicable_rows else ""
                ),
                "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else "",
                "avg_source_coverage_ratio": _avg_optional(
                    row.source_coverage_ratio for row in method_rows
                ),
                "total_prompt_tokens": _sum_optional(row.prompt_tokens for row in method_rows),
                "total_output_tokens": _sum_optional(row.output_tokens for row in method_rows),
                "total_thought_tokens": _sum_optional(row.thought_tokens for row in method_rows),
                "total_tokens": _sum_optional(row.total_tokens for row in method_rows),
                "estimated_cost_usd": _round_optional(
                    _sum_optional(row.estimated_cost_usd for row in method_rows)
                ),
            }
        )
    return summaries


def print_overview(rows: list[EvaluationRow]) -> None:
    methods = sorted({row.retrieval_method for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row.retrieval_method == method]
        applicable_rows = [row for row in method_rows if row.retrieval_check_applicable]
        total = len(method_rows)
        errors = sum(1 for row in method_rows if row.error)
        section_hits = sum(1 for row in applicable_rows if row.expected_section_in_top_3)
        avg_latency = sum(row.latency_seconds for row in method_rows) / max(total, 1)
        print(
            f"{method}: {section_hits}/{len(applicable_rows)} expected sections in top 3, "
            f"{errors} errors, avg {avg_latency:.2f}s"
        )


def _sum_optional(values: Any) -> float | int | str:
    numeric_values = [value for value in values if isinstance(value, (int, float))]
    if not numeric_values:
        return ""

    total = sum(numeric_values)
    return int(total) if float(total).is_integer() else total


def _round_optional(value: float | int | str) -> float | str:
    if isinstance(value, (int, float)):
        return round(float(value), 8)

    return ""


def _avg_optional(values: Any) -> float | str:
    numeric_values = [value for value in values if isinstance(value, (int, float))]
    if not numeric_values:
        return ""

    return round(sum(numeric_values) / len(numeric_values), 4)


if __name__ == "__main__":
    main()
