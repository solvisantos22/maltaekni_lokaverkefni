"""FastAPI app for the Icelandic consumer-rights RAG interface."""

from __future__ import annotations

import csv
import json
import os
import re
import secrets
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer_generator import generate_grounded_answer
from .retriever import build_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
EVALUATION_DIR = PROJECT_ROOT / "reports" / "evaluation"
EVALUATION_SUMMARY_PATH = EVALUATION_DIR / "evaluation_summary_latest.csv"
EVALUATION_DETAILS_PATH = EVALUATION_DIR / "evaluation_details_latest.jsonl"
EVALUATION_REVIEW_PATH = EVALUATION_DIR / "evaluation_review_latest.csv"
EVALUATION_REVIEW_GLOB = "evaluation_review_*.csv"
DEMO_EVALUATION_DIR = PROJECT_ROOT / "docs" / "demo_evaluation"
DEMO_EVALUATION_SUMMARY_PATH = DEMO_EVALUATION_DIR / "evaluation_summary_demo.csv"
DEMO_EVALUATION_DETAILS_PATH = DEMO_EVALUATION_DIR / "evaluation_details_demo.jsonl"
DEMO_EVALUATION_REVIEW_PATH = DEMO_EVALUATION_DIR / "evaluation_review_demo.csv"
WEB_DIR = Path(__file__).resolve().parent / "web"
ACCESS_TOKEN_ENV = "APP_ACCESS_TOKEN"

app = FastAPI(title="Réttarvísir")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    method: Literal[
        "tfidf",
        "bm25",
        "icebert",
        "bge-m3",
        "rrf-icebert-bm25",
        "rrf-bge-m3-bm25",
        "rrf-bge-m3-bm25-rerank",
    ] = "tfidf"
    top_k: int = Field(default=3, ge=1, le=5)


class StatusResponse(BaseModel):
    ready: bool
    chunks_path: str
    message: str
    access_required: bool = False


class EvaluationReviewRequest(BaseModel):
    question_id: str
    retrieval_method: str
    evaluator: str = Field(min_length=1, max_length=80)
    retrieval_relevance_1_5: int | None = Field(default=None, ge=1, le=5)
    answer_correctness_1_5: int | None = Field(default=None, ge=1, le=5)
    source_support_1_5: int | None = Field(default=None, ge=1, le=5)
    clarity_1_5: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=2000)


@lru_cache(maxsize=8)
def _load_retriever(method: str):
    return build_retriever(method, chunks_path=CHUNKS_PATH)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/evaluation")
def evaluation_review():
    return FileResponse(WEB_DIR / "evaluation.html")


@app.get("/evaluation/dashboard")
def evaluation_dashboard():
    return FileResponse(WEB_DIR / "evaluation_dashboard.html")


@app.get("/api/status", response_model=StatusResponse)
def status():
    if CHUNKS_PATH.exists():
        return StatusResponse(
            ready=True,
            chunks_path=str(CHUNKS_PATH),
            message="ready",
            access_required=access_token_required(),
        )

    return StatusResponse(
        ready=False,
        chunks_path=str(CHUNKS_PATH),
        access_required=access_token_required(),
        message=(
            "Run: python src\\maltaekni_lokaverkefni\\fetch_sources.py "
            "and python src\\maltaekni_lokaverkefni\\chunking.py"
        ),
    )


@app.post("/api/ask")
def ask(
    request: AskRequest,
    x_app_access_token: str | None = Header(default=None),
):
    verify_access_token(x_app_access_token)
    if not CHUNKS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Missing data/processed/chunks.json. Run fetch_sources.py "
                "and chunking.py before starting the app."
            ),
        )

    retriever = _load_retriever(request.method)
    retrieval_result = retriever.search(request.question.strip(), top_k=request.top_k)
    answer_result = generate_grounded_answer(
        retrieval_result,
        max_sources=request.top_k,
    )
    return answer_result.to_dict()


def access_token_required() -> bool:
    """Return whether deployed users must provide the shared access token."""
    return bool(os.getenv(ACCESS_TOKEN_ENV, "").strip())


def verify_access_token(provided_token: str | None) -> None:
    """Reject costly endpoints when APP_ACCESS_TOKEN is configured."""
    expected_token = os.getenv(ACCESS_TOKEN_ENV, "").strip()
    if not expected_token:
        return

    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(
            status_code=401,
            detail="Access token is missing or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/api/evaluation/latest")
def latest_evaluation(demo: bool = False):
    paths = _evaluation_paths(demo=demo)
    if not paths["summary"].exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Missing reports/evaluation/evaluation_summary_latest.csv. "
                "Run evaluate_methods.py first or open /evaluation?demo=1."
            ),
        )

    rows = _load_evaluation_rows(paths["summary"])
    details = _load_evaluation_details(paths["details"])
    reviews = _load_evaluation_reviews(paths["review"])

    for row in rows:
        key = _review_key(row["question_id"], row["retrieval_method"])
        row["row_key"] = key
        row["sources"] = details.get(key, {}).get("answer_result", {}).get("sources", [])

    return {
        "mode": paths["mode"],
        "is_demo": paths["mode"] == "demo",
        "summary_path": str(paths["summary"]),
        "details_path": str(paths["details"]),
        "review_path": _format_review_paths(paths["review"]),
        "rows": rows,
        "reviews": reviews,
    }


@app.get("/api/evaluation/dashboard")
def evaluation_dashboard_data(demo: bool = False):
    paths = _evaluation_paths(demo=demo)
    if not paths["summary"].exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Missing reports/evaluation/evaluation_summary_latest.csv. "
                "Run evaluate_methods.py first or open /evaluation/dashboard?demo=1."
            ),
        )

    rows = _load_evaluation_rows(paths["summary"])
    review_rows = _load_evaluation_review_rows(paths["review"])
    return {
        "mode": paths["mode"],
        "is_demo": paths["mode"] == "demo",
        "summary_path": str(paths["summary"]),
        "review_path": _format_review_paths(paths["review"]),
        "overall": _dashboard_overall(rows, review_rows),
        "methods": _dashboard_by_method(rows, review_rows),
    }


@app.post("/api/evaluation/review")
def save_evaluation_review(review: EvaluationReviewRequest):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    review_path = _review_path_for_evaluator(review.evaluator)
    reviews = _load_evaluation_review_rows(review_path)
    review_row = review.model_dump()
    review_row["row_key"] = _review_key(review.question_id, review.retrieval_method)

    existing_index = next(
        (
            index
            for index, row in enumerate(reviews)
            if row.get("row_key") == review_row["row_key"]
            and row.get("evaluator") == review.evaluator
        ),
        None,
    )
    if existing_index is None:
        reviews.append(review_row)
    else:
        reviews[existing_index] = review_row

    _write_evaluation_reviews(reviews, review_path)
    return {"saved": True, "review": review_row, "review_path": str(review_path)}


def _evaluation_paths(*, demo: bool = False) -> dict[str, Path | list[Path] | str]:
    if demo or not EVALUATION_SUMMARY_PATH.exists():
        return {
            "mode": "demo",
            "summary": DEMO_EVALUATION_SUMMARY_PATH,
            "details": DEMO_EVALUATION_DETAILS_PATH,
            "review": DEMO_EVALUATION_REVIEW_PATH,
        }

    return {
        "mode": "real",
        "summary": EVALUATION_SUMMARY_PATH,
        "details": EVALUATION_DETAILS_PATH,
        "review": _evaluation_review_paths(),
    }


def _load_evaluation_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["answer"] = row.get("answer", "").replace("||", "\n\n")
    return rows


def _load_evaluation_details(path: Path) -> dict[str, dict]:
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
            details[_review_key(question_id, method)] = item
    return details


def _evaluation_review_paths() -> list[Path]:
    review_paths = sorted(
        path
        for path in EVALUATION_DIR.glob(EVALUATION_REVIEW_GLOB)
        if path.name != EVALUATION_REVIEW_PATH.name
    )
    if review_paths:
        return review_paths
    if EVALUATION_REVIEW_PATH.exists():
        return [EVALUATION_REVIEW_PATH]
    return []


def _review_path_for_evaluator(evaluator: str) -> Path:
    return EVALUATION_DIR / f"evaluation_review_{_evaluator_slug(evaluator)}.csv"


def _evaluator_slug(evaluator: str) -> str:
    normalized = unicodedata.normalize("NFKD", evaluator).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    return slug or "unknown"


def _format_review_paths(path: Path | list[Path]) -> str:
    if isinstance(path, list):
        if not path:
            return str(EVALUATION_DIR / EVALUATION_REVIEW_GLOB)
        return ", ".join(str(item) for item in path)
    return str(path)


def _load_evaluation_reviews(
    path: Path | list[Path] | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    reviews: dict[str, dict[str, dict[str, str]]] = {}
    for row in _load_evaluation_review_rows(path):
        row_key = row.get("row_key", "")
        evaluator = row.get("evaluator", "")
        if not row_key or not evaluator:
            continue
        reviews.setdefault(row_key, {})[evaluator] = row
    return reviews


def _load_evaluation_review_rows(
    path: Path | list[Path] | None = None,
) -> list[dict[str, str]]:
    paths = _evaluation_review_paths() if path is None else path
    if isinstance(paths, Path):
        paths = [paths]

    rows = []
    for review_path in paths:
        if not review_path.exists():
            continue
        with review_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def _write_evaluation_reviews(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "row_key",
        "question_id",
        "retrieval_method",
        "evaluator",
        "retrieval_relevance_1_5",
        "answer_correctness_1_5",
        "source_support_1_5",
        "clarity_1_5",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _dashboard_overall(
    rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
) -> dict[str, Any]:
    total = len(rows)
    return {
        "rows": total,
        "methods": len({row.get("retrieval_method", "") for row in rows if row.get("retrieval_method")}),
        "questions": len({row.get("question_id", "") for row in rows if row.get("question_id")}),
        "retrieval_check_questions": sum(1 for row in rows if _retrieval_check_applicable(row)),
        "run_labels": sorted({row.get("run_label", "") for row in rows if row.get("run_label")}),
        "prompt_profiles": sorted(
            {row.get("prompt_profile", "") for row in rows if row.get("prompt_profile")}
        ),
        "errors": sum(1 for row in rows if row.get("error")),
        "review_count": len(review_rows),
        "total_tokens": _sum_number(row.get("total_tokens") for row in rows),
        "estimated_cost_usd": _round_number(
            _sum_number(row.get("estimated_cost_usd") for row in rows)
        ),
    }


def _dashboard_by_method(
    rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    methods = sorted({row.get("retrieval_method", "") for row in rows if row.get("retrieval_method")})
    method_reviews: dict[str, list[dict[str, str]]] = {}
    for review in review_rows:
        method = review.get("retrieval_method", "")
        if method:
            method_reviews.setdefault(method, []).append(review)

    summaries = []
    for method in methods:
        method_rows = [row for row in rows if row.get("retrieval_method") == method]
        applicable_rows = [row for row in method_rows if _retrieval_check_applicable(row)]
        reviews = method_reviews.get(method, [])
        total = len(method_rows)
        expected_hits = sum(
            1
            for row in applicable_rows
            if str(row.get("expected_section_in_top_3", "")).lower() == "true"
        )
        summaries.append(
            {
                "retrieval_method": method,
                "run_label": _first_value(row.get("run_label", "") for row in method_rows),
                "prompt_profile": _first_value(row.get("prompt_profile", "") for row in method_rows),
                "llm_model": _first_value(row.get("llm_model", "") for row in method_rows),
                "rows": total,
                "retrieval_check_questions": len(applicable_rows),
                "errors": sum(1 for row in method_rows if row.get("error")),
                "expected_section_top3_hits": expected_hits,
                "expected_section_top3_rate": (
                    round(expected_hits / len(applicable_rows), 4) if applicable_rows else None
                ),
                "avg_source_coverage_ratio": _avg_number(
                    row.get("source_coverage_ratio") for row in method_rows
                ),
                "avg_latency_seconds": _avg_number(row.get("latency_seconds") for row in method_rows),
                "total_prompt_tokens": _sum_number(row.get("prompt_tokens") for row in method_rows),
                "total_output_tokens": _sum_number(row.get("output_tokens") for row in method_rows),
                "total_thought_tokens": _sum_number(row.get("thought_tokens") for row in method_rows),
                "total_tokens": _sum_number(row.get("total_tokens") for row in method_rows),
                "estimated_cost_usd": _round_number(
                    _sum_number(row.get("estimated_cost_usd") for row in method_rows)
                ),
                "human_reviews": len(reviews),
                "avg_retrieval_relevance_1_5": _avg_number(
                    review.get("retrieval_relevance_1_5") for review in reviews
                ),
                "avg_answer_correctness_1_5": _avg_number(
                    review.get("answer_correctness_1_5") for review in reviews
                ),
                "avg_source_support_1_5": _avg_number(
                    review.get("source_support_1_5") for review in reviews
                ),
                "avg_clarity_1_5": _avg_number(review.get("clarity_1_5") for review in reviews),
            }
        )
    return summaries


def _sum_number(values: Any) -> float | int:
    numbers = [_to_number(value) for value in values]
    valid_numbers = [value for value in numbers if value is not None]
    total = sum(valid_numbers)
    return int(total) if float(total).is_integer() else total


def _avg_number(values: Any) -> float | None:
    numbers = [_to_number(value) for value in values]
    valid_numbers = [value for value in numbers if value is not None]
    if not valid_numbers:
        return None

    return round(sum(valid_numbers) / len(valid_numbers), 3)


def _round_number(value: float | int) -> float:
    return round(float(value), 8)


def _to_number(value: Any) -> float | None:
    if value in {None, ""}:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _retrieval_check_applicable(row: dict[str, str]) -> bool:
    explicit = str(row.get("retrieval_check_applicable", "")).lower()
    if explicit in {"true", "false"}:
        return explicit == "true"

    expected_section = " ".join(row.get("expected_relevant_section", "").casefold().split())
    return bool(expected_section) and expected_section not in {"no_relevant_source", "none", "n/a"}


def _first_value(values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _review_key(question_id: str, retrieval_method: str) -> str:
    return f"{question_id}::{retrieval_method}"


def main():
    import uvicorn

    uvicorn.run(
        "src.maltaekni_lokaverkefni.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
