"""FastAPI app for the Icelandic consumer-rights RAG interface."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .answer_generator import generate_grounded_answer
    from .retriever import build_retriever
except ImportError:  # Allows direct script execution during early experiments.
    from answer_generator import generate_grounded_answer
    from retriever import build_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Neytendaréttur RAG")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    method: Literal["tfidf", "bm25"] = "tfidf"
    top_k: int = Field(default=3, ge=1, le=5)


class StatusResponse(BaseModel):
    ready: bool
    chunks_path: str
    message: str


@lru_cache(maxsize=4)
def _load_retriever(method: str):
    return build_retriever(method, chunks_path=CHUNKS_PATH)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


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
