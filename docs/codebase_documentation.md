# Codebase Documentation

This document explains the main parts of the project so an external reader can
understand how the system works without reading every implementation detail.
The project is a local Icelandic consumer-rights RAG prototype: it retrieves
legal source chunks, generates grounded Icelandic answers, shows citations in a
web UI, and records evaluation results for the final report.

## End-to-End Flow

1. `fetch_sources.py` downloads curated Althingi Lagasafn pages and writes
   cleaned page-level `Document` records to `data/processed/documents.json`.
2. `chunking.py` splits each law into article-level `Chunk` records and writes
   `data/processed/chunks.json`. It also writes `chunk_lemmas.json`, a cache of
   Icelandic lemmas used by lexical retrieval.
3. `retriever.py` loads chunks and ranks them for a user question with one of
   the supported retrieval methods.
4. `answer_generator.py` builds a grounded prompt from the top chunks and uses
   Gemini/OpenAI when configured. If no LLM key is available, it uses a local
   extractive fallback.
5. `app.py` exposes the chat, evaluation dashboard, and manual review UI with
   FastAPI.
6. `evaluate_methods.py` runs the fixed question set through selected retrieval
   methods and writes automatic evaluation artifacts.
7. `export_report_tables.py` turns saved evaluation and human-review files into
   report-ready CSV tables.

## Source Data

The source fetcher currently supports curated Althingi law pages. The parser
extracts the law body from Lagasafn HTML, removes navigation and amendment
noise, and preserves article headings so later citations can point back to a
specific legal section.

The processed data files are generated artifacts:

- `data/processed/documents.json`: cleaned page-level laws.
- `data/processed/chunks.json`: article-level retrieval chunks.
- `data/processed/chunk_lemmas.json`: cached lemmatized searchable text.
- `data/processed/embedding_cache/*.npz`: optional embedding caches keyed by
  chunk fingerprint.

## Core Data Shapes

The shared dataclasses live in `types_classes.py`.

- `Document`: one cleaned legal source page before chunking.
- `Chunk`: one searchable legal article with title, URL, section, source, and
  text.
- `EvaluationRow`: one flattened question/method result written to evaluation
  CSV files.

Keeping these shapes small is intentional. It makes the fetcher, chunker,
retrievers, answer generator, UI, and evaluation scripts communicate through
stable contracts.

## Retrieval Methods

`retriever.py` exposes one `Retriever` class with a method switch. All methods
return the same contract:

```json
{
  "question": "...",
  "chunks": [
    {
      "chunk_id": "...",
      "text": "...",
      "source": "althingi",
      "title": "...",
      "url": "...",
      "section": "...",
      "score": 0.0,
      "retrieval_method": "..."
    }
  ]
}
```

Supported methods:

- `tfidf`: lexical TF-IDF over Icelandic lemmas.
- `bm25`: BM25 over Icelandic lemmas.
- `icebert`: direct embedding retrieval with IceBERT.
- `bge-m3`: direct embedding retrieval with BGE-M3.
- `rrf-icebert-bm25`: reciprocal rank fusion between BM25 and IceBERT.
- `rrf-bge-m3-bm25`: reciprocal rank fusion between BM25 and BGE-M3.
- `rrf-bge-m3-bm25-rerank`: BGE-M3/BM25 fusion followed by cross-encoder
  reranking.

`ice_tokenizer.py` wraps `tokenizer` and Reynir so lexical methods can compare
lemmatized forms instead of relying only on exact surface words.

## Answer Generation

`prompts.py` defines the answer prompt templates and prompt profiles. The
generator is designed to answer only from retrieved chunks and to include
numbered citations such as `[1]`.

`answer_generator.py` chooses the answer path:

- Gemini if `LLM_PROVIDER=gemini` or `auto` and `GEMINI_API_KEY` is available.
- OpenAI if `LLM_PROVIDER=openai` or Gemini is unavailable in `auto` mode.
- Local extractive fallback if no LLM provider is configured or the provider
  call fails.

The returned `AnswerResult` stores the answer, cited sources, prompts,
confidence label, source coverage, provider name, model name, token counts, and
optional estimated cost.

## Web App

`app.py` is the FastAPI entry point.

- `/`: main chat UI.
- `/api/ask`: retrieval plus grounded answer generation.
- `/api/status`: checks whether the processed chunks are available.
- `/evaluation`: manual review UI for saved evaluation rows.
- `/evaluation/dashboard`: dashboard for automatic and human evaluation metrics.
- `/api/evaluation/latest`: saved evaluation rows and details for the review UI.
- `/api/evaluation/review`: saves human review scores to evaluator-specific CSVs.
- `/api/evaluation/dashboard`: aggregate metrics for the dashboard.

Static UI files live in `src/maltaekni_lokaverkefni/web/`:

- `index.html`, `app.js`, `styles.css`: main chat experience.
- `evaluation.html`: manual review interface.
- `evaluation_dashboard.html`: saved-results dashboard.
- `teacher_guide.js`, `teacher_guide.css`: movable teacher checklist.

## Evaluation

The fixed evaluation question set is `docs/evaluation_questions.csv`. It
includes legal-term questions and everyday paraphrases so the report can discuss
how sensitive each retrieval method is to legal vocabulary.

`evaluate_methods.py` writes three main artifact types under
`reports/evaluation/`:

- `evaluation_summary_*.csv`: flattened rows for automatic metrics.
- `evaluation_details_*.jsonl`: full retrieval and answer traces.
- `evaluation_method_summary_*.csv`: aggregate automatic metrics by retrieval
  method.

The automatic top-3 source metric checks whether the expected legal section from
the question CSV appears in the retrieved top three sections. Questions marked
as no-source or not applicable are excluded from that retrieval hit-rate
calculation.

Manual review is separate from the automatic run. Each evaluator scores the same
saved answers in the UI on four 1-5 dimensions:

- retrieval relevance
- answer correctness
- source support
- clarity

The saved review files are intended to be committed separately, for example
`evaluation_review_solvi.csv` and `evaluation_review_johannes.csv`. After both
review files exist, `export_report_tables.py` produces the final CSV tables for
the report.

## Main Commands

Generate source documents and chunks:

```powershell
python -m src.maltaekni_lokaverkefni.fetch_sources
python -m src.maltaekni_lokaverkefni.chunking
```

Start the app:

```powershell
python -m uvicorn src.maltaekni_lokaverkefni.app:app --host 127.0.0.1 --port 8000
```

Run a no-token smoke evaluation:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --no-llm --limit 2
```

Export report tables:

```powershell
python -m src.maltaekni_lokaverkefni.export_report_tables
```

External review uses the same local FastAPI app. A temporary Gemini key is
provided privately for review and placed in the reviewer's local `.env` file.
The key is not committed to Git and is revoked after review.

## Design Notes

- The system keeps original legal text for display and citations, while
  retrieval can use normalized or embedded representations internally.
- Retrieval and answer generation are deliberately separated so the report can
  compare retrieval methods under the same answer-generation setup.
- Evaluation artifacts are saved before human review, so manual scoring never
  triggers new LLM calls.
