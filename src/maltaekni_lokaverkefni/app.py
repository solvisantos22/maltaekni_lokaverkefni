"""FastAPI app for the Icelandic consumer-rights RAG interface."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
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
WEB_DIR = Path(__file__).resolve().parent / "web"

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
    ] = "tfidf"
    top_k: int = Field(default=3, ge=1, le=5)


class StatusResponse(BaseModel):
    ready: bool
    chunks_path: str
    message: str


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


@app.get("/api/status", response_model=StatusResponse)
def status():
    if CHUNKS_PATH.exists():
        return StatusResponse(
            ready=True,
            chunks_path=str(CHUNKS_PATH),
            message="ready",
        )

    return StatusResponse(
        ready=False,
        chunks_path=str(CHUNKS_PATH),
        message=(
            "Run: python src\\maltaekni_lokaverkefni\\fetch_sources.py "
            "and python src\\maltaekni_lokaverkefni\\chunking.py"
        ),
    )


@app.post("/api/ask")
def ask(request: AskRequest):
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


@app.get("/api/evaluation/latest")
def latest_evaluation():
    if not EVALUATION_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Missing reports/evaluation/evaluation_summary_latest.csv. "
                "Run evaluate_methods.py first."
            ),
        )

    rows = _load_evaluation_rows()
    details = _load_evaluation_details()
    reviews = _load_evaluation_reviews()

    for row in rows:
        key = _review_key(row["question_id"], row["retrieval_method"])
        row["row_key"] = key
        row["sources"] = details.get(key, {}).get("answer_result", {}).get("sources", [])

    return {
        "summary_path": str(EVALUATION_SUMMARY_PATH),
        "details_path": str(EVALUATION_DETAILS_PATH),
        "review_path": str(EVALUATION_REVIEW_PATH),
        "rows": rows,
        "reviews": reviews,
    }


@app.post("/api/evaluation/review")
def save_evaluation_review(review: EvaluationReviewRequest):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    reviews = _load_evaluation_review_rows()
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

    _write_evaluation_reviews(reviews)
    return {"saved": True, "review": review_row}


def _load_evaluation_rows() -> list[dict[str, str]]:
    with EVALUATION_SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_evaluation_details() -> dict[str, dict]:
    if not EVALUATION_DETAILS_PATH.exists():
        return {}

    details = {}
    with EVALUATION_DETAILS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question", {}).get("id", ""))
            method = str(item.get("retrieval_method", ""))
            details[_review_key(question_id, method)] = item
    return details


def _load_evaluation_reviews() -> dict[str, dict[str, dict[str, str]]]:
    reviews: dict[str, dict[str, dict[str, str]]] = {}
    for row in _load_evaluation_review_rows():
        row_key = row.get("row_key", "")
        evaluator = row.get("evaluator", "")
        if not row_key or not evaluator:
            continue
        reviews.setdefault(row_key, {})[evaluator] = row
    return reviews


def _load_evaluation_review_rows() -> list[dict[str, str]]:
    if not EVALUATION_REVIEW_PATH.exists():
        return []

    with EVALUATION_REVIEW_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_evaluation_reviews(rows: list[dict]) -> None:
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
    with EVALUATION_REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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
