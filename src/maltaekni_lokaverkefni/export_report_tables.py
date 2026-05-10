"""Export report-ready CSV tables from saved evaluation outputs.

This script is intentionally standalone: it only reads CSV/JSONL artifacts and
does not import the app, retriever, or model code. That makes it safe to run on a
machine that only needs to assemble report tables after evaluation has finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "reports" / "evaluation"
DEFAULT_OUTPUT_DIR = DEFAULT_EVALUATION_DIR / "report_tables"

SCORE_FIELDS = [
    "retrieval_relevance_1_5",
    "answer_correctness_1_5",
    "source_support_1_5",
    "clarity_1_5",
]


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Export report-ready evaluation tables.")
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
        help="Directory containing evaluation_summary_latest.csv and review CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where report tables will be written.",
    )
    args = parser.parse_args()

    summary_path = args.evaluation_dir / "evaluation_summary_latest.csv"
    details_path = args.evaluation_dir / "evaluation_details_latest.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    rows = read_csv(summary_path)
    review_rows = read_review_rows(args.evaluation_dir)
    details = read_details(details_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table_retrieval_methods.csv": build_retrieval_table(rows),
        "table_cost_latency.csv": build_cost_latency_table(rows),
        "table_human_scores.csv": build_human_scores_table(review_rows),
        "table_inter_reviewer.csv": build_inter_reviewer_table(review_rows),
        "qualitative_cases.csv": build_qualitative_cases(rows, review_rows, details),
    }

    for filename, table_rows in outputs.items():
        write_csv(args.output_dir / filename, table_rows)
        print(f"Wrote {args.output_dir / filename}")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV artifact as dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_review_rows(evaluation_dir: Path) -> list[dict[str, str]]:
    """Load all committed human review CSVs except the legacy latest file."""
    paths = sorted(
        path
        for path in evaluation_dir.glob("evaluation_review_*.csv")
        if path.name != "evaluation_review_latest.csv"
    )
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv(path))
    return rows


def read_details(path: Path) -> dict[str, dict[str, Any]]:
    """Load JSONL evaluation traces keyed by question id and retrieval method."""
    if not path.exists():
        return {}

    details = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question", {}).get("id", ""))
            method = str(item.get("retrieval_method", ""))
            if question_id and method:
                details[row_key(question_id, method)] = item
    return details


def build_retrieval_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Aggregate automatic top-3 retrieval metrics by method."""
    table = []
    for method, method_rows in grouped(rows, "retrieval_method").items():
        applicable = [row for row in method_rows if is_true(row.get("retrieval_check_applicable"))]
        hits = sum(1 for row in applicable if is_true(row.get("expected_section_in_top_3")))
        table.append(
            {
                "retrieval_method": method,
                "questions": len(method_rows),
                "retrieval_check_questions": len(applicable),
                "expected_section_top3_hits": hits,
                "expected_section_top3_rate": ratio(hits, len(applicable)),
                "avg_source_coverage_ratio": avg(row.get("source_coverage_ratio") for row in method_rows),
                "avg_source_coverage_ratio_answered": avg(
                    row.get("source_coverage_ratio")
                    for row in method_rows
                    if not is_abstention(row)
                ),
                "errors": sum(1 for row in method_rows if row.get("error")),
            }
        )
    return sorted(table, key=lambda row: row["retrieval_method"])


def build_cost_latency_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Aggregate runtime, token, and optional cost metrics by method."""
    table = []
    for method, method_rows in grouped(rows, "retrieval_method").items():
        table.append(
            {
                "retrieval_method": method,
                "run_label": first_value(row.get("run_label") for row in method_rows),
                "prompt_profile": first_value(row.get("prompt_profile") for row in method_rows),
                "llm_model": first_value(row.get("llm_model") for row in method_rows),
                "avg_latency_seconds": avg(row.get("latency_seconds") for row in method_rows),
                "total_prompt_tokens": total(row.get("prompt_tokens") for row in method_rows),
                "total_output_tokens": total(row.get("output_tokens") for row in method_rows),
                "total_thought_tokens": total(row.get("thought_tokens") for row in method_rows),
                "total_tokens": total(row.get("total_tokens") for row in method_rows),
                "estimated_cost_usd": total(estimated_cost(row) for row in method_rows),
            }
        )
    return sorted(table, key=lambda row: row["retrieval_method"])


def build_human_scores_table(review_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Aggregate manual 1-5 review scores by retrieval method."""
    table = []
    for method, method_reviews in grouped(review_rows, "retrieval_method").items():
        row = {
            "retrieval_method": method,
            "review_count": len(method_reviews),
        }
        for field in SCORE_FIELDS:
            row[f"avg_{field}"] = avg(review.get(field) for review in method_reviews)
        row["overall_human_avg"] = avg(
            review.get(field) for review in method_reviews for field in SCORE_FIELDS
        )
        table.append(row)
    return sorted(table, key=lambda row: row["retrieval_method"])


def build_inter_reviewer_table(review_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare per-method average scores between evaluators."""
    by_method_evaluator: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for review in review_rows:
        by_method_evaluator[review.get("retrieval_method", "")][review.get("evaluator", "")].append(review)

    table = []
    for method, by_evaluator in by_method_evaluator.items():
        row: dict[str, Any] = {"retrieval_method": method}
        evaluator_scores = {}
        for evaluator, evaluator_rows in sorted(by_evaluator.items()):
            evaluator_slug = slug(evaluator)
            score = avg(review.get(field) for review in evaluator_rows for field in SCORE_FIELDS)
            evaluator_scores[evaluator] = score
            row[f"{evaluator_slug}_review_count"] = len(evaluator_rows)
            row[f"{evaluator_slug}_overall_avg"] = score
        if len(evaluator_scores) == 2:
            values = list(evaluator_scores.values())
            row["absolute_avg_difference"] = round(abs(values[0] - values[1]), 4)
        table.append(row)
    return sorted(table, key=lambda row: row["retrieval_method"])


def build_qualitative_cases(
    rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    details: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick representative strong, weak, miss, and error cases for discussion."""
    review_scores = build_review_score_index(review_rows)
    enriched = []
    for row in rows:
        key = row_key(row.get("question_id", ""), row.get("retrieval_method", ""))
        human_avg = review_scores.get(key)
        enriched.append((row, key, human_avg))

    cases = [
        pick_case(
            "strong_answer",
            enriched,
            quality_score,
            reverse=True,
        ),
        pick_case(
            "weak_answer",
            enriched,
            quality_score,
            reverse=False,
        ),
        pick_case(
            "retrieval_hit_but_answer_weak",
            [item for item in enriched if is_true(item[0].get("expected_section_in_top_3"))],
            quality_score,
            reverse=False,
        ),
        pick_case(
            "retrieval_miss",
            [item for item in enriched if item[0].get("retrieval_check_applicable") and not is_true(item[0].get("expected_section_in_top_3"))],
            lambda item: float_or_none(item[0].get("top_1_score")) or 0,
            reverse=True,
        ),
        pick_case(
            "error_case",
            [item for item in enriched if item[0].get("error")],
            lambda item: item[0].get("question_id", ""),
            reverse=False,
        ),
    ]
    return [format_case(case, details) for case in cases if case is not None]


def quality_score(item: tuple[dict[str, str], str, float | None]) -> tuple[float, float, float]:
    row, _, human_avg = item
    if human_avg is not None:
        return (human_avg, float_or_none(row.get("source_coverage_ratio")) or 0, 0)

    top3_bonus = 1.0 if is_true(row.get("expected_section_in_top_3")) else 0.0
    no_error_bonus = 1.0 if not row.get("error") else 0.0
    coverage = float_or_none(row.get("source_coverage_ratio")) or 0.0
    return (top3_bonus + no_error_bonus + coverage, coverage, float_or_none(row.get("top_1_score")) or 0)


def build_review_score_index(review_rows: list[dict[str, str]]) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for review in review_rows:
        key = row_key(review.get("question_id", ""), review.get("retrieval_method", ""))
        for field in SCORE_FIELDS:
            value = float_or_none(review.get(field))
            if value is not None:
                scores[key].append(value)
    return {key: round(sum(values) / len(values), 4) for key, values in scores.items() if values}


def pick_case(label: str, items: list[tuple[dict[str, str], str, float | None]], key, *, reverse: bool):
    if not items:
        return None
    row, row_key_value, human_avg = sorted(items, key=key, reverse=reverse)[0]
    return label, row, row_key_value, human_avg


def format_case(case, details: dict[str, dict[str, Any]]) -> dict[str, Any]:
    label, row, key, human_avg = case
    sources = details.get(key, {}).get("answer_result", {}).get("sources", [])
    return {
        "case_type": label,
        "question_id": row.get("question_id", ""),
        "retrieval_method": row.get("retrieval_method", ""),
        "question": row.get("question", ""),
        "expected_relevant_section": row.get("expected_relevant_section", ""),
        "expected_section_in_top_3": row.get("expected_section_in_top_3", ""),
        "human_avg": human_avg if human_avg is not None else "",
        "top_1_section": row.get("top_1_section", ""),
        "source_coverage_ratio": row.get("source_coverage_ratio", ""),
        "answer_excerpt": excerpt(row.get("answer", "")),
        "first_source_excerpt": excerpt(sources[0].get("text", "") if sources else row.get("top_3_sections", "")),
        "error": row.get("error", ""),
    }


def grouped(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(field, "")].append(row)
    return groups


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(question_id: str, method: str) -> str:
    return f"{question_id}::{method}"


def is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def is_abstention(row: dict[str, str]) -> bool:
    """Return whether the answer explicitly says the sources are insufficient."""
    return "ég finn ekki nægar upplýsingar" in row.get("answer", "").casefold()


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: Any) -> float | str:
    numbers = [value for value in (float_or_none(value) for value in values) if value is not None]
    return round(sum(numbers) / len(numbers), 4) if numbers else ""


def total(values: Any) -> float | int | str:
    numbers = [value for value in (float_or_none(value) for value in values) if value is not None]
    if not numbers:
        return ""
    summed = sum(numbers)
    return int(summed) if summed.is_integer() else round(summed, 8)


def estimated_cost(row: dict[str, str]) -> float | str:
    """Use saved cost when present, otherwise estimate from token counts and env rates."""
    saved_cost = float_or_none(row.get("estimated_cost_usd"))
    if saved_cost is not None:
        return saved_cost

    provider = (row.get("llm_provider") or "").strip().upper()
    if not provider:
        return ""

    input_rate = float_or_none(os.getenv(f"{provider}_INPUT_COST_PER_1M"))
    output_rate = float_or_none(os.getenv(f"{provider}_OUTPUT_COST_PER_1M"))
    if input_rate is None and output_rate is None:
        return ""

    prompt_tokens = float_or_none(row.get("prompt_tokens")) or 0
    output_tokens = float_or_none(row.get("output_tokens")) or 0
    thought_tokens = float_or_none(row.get("thought_tokens")) or 0
    input_cost = (prompt_tokens / 1_000_000) * (input_rate or 0)
    output_cost = ((output_tokens + thought_tokens) / 1_000_000) * (output_rate or 0)
    return round(input_cost + output_cost, 8)


def ratio(numerator: int, denominator: int) -> float | str:
    return round(numerator / denominator, 4) if denominator else ""


def first_value(values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def slug(value: str) -> str:
    return (
        value.casefold()
        .replace("ö", "o")
        .replace("ó", "o")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ú", "u")
        .replace("ý", "y")
        .replace("ð", "d")
        .replace("þ", "th")
    )


def excerpt(text: str, limit: int = 260) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


if __name__ == "__main__":
    main()
