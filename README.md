# Máltækni lokaverkefni

Final NLP project for TÖL025M Inngangur að máltækni.

Authors: Sölvi and Jóhannes

## Overview

Réttarvísir is a local Icelandic consumer-rights RAG prototype. It retrieves
legal source chunks, generates grounded Icelandic answers with citations, and
includes saved-results evaluation and review screens that do not trigger new
LLM calls.

## Structure

- `src/maltaekni_lokaverkefni/`: application, retrieval, answer generation,
  evaluation, and report-export code.
- `src/maltaekni_lokaverkefni/web/`: static HTML, CSS, and JavaScript for the
  chat UI, evaluation dashboard, and manual review screens.
- `docs/`: final technical documentation, evaluation protocol, evaluation
  questions, methodology notes, and teacher-facing run instructions.
- `reports/evaluation/`: saved automatic evaluation outputs, human review CSVs,
  and exported report tables.
- `data/processed/`: regenerated source documents, chunks, lemma caches, and
  embedding caches.
- `data/raw`: list containing icelandic stopwords

## Documentation

The main technical documentation is in `docs/codebase_documentation.md`. It
explains the full pipeline, module responsibilities, generated artifacts,
retrieval methods, answer generation, web endpoints, and evaluation workflow.

Teacher-facing local run instructions can be generated locally from the
maintainers' Markdown handout and shared outside Git as a PDF.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Local App

```powershell
python -m uvicorn src.maltaekni_lokaverkefni.app:app --host 127.0.0.1 --port 8000
```

Main URL:

```text
http://127.0.0.1:8000
```

Direct chat URL without the welcome screen:

```text
http://127.0.0.1:8000/?skipWelcome=1
```

The main app sidebar links to the evaluation dashboard and human review screens.
There you can see the evaluated cases.

```text
http://127.0.0.1:8000/evaluation
http://127.0.0.1:8000/evaluation/dashboard
```

Live retrieval requires `data/processed/chunks.json`. The source and chunk files
are regenerated with:

```powershell
python -m src.maltaekni_lokaverkefni.fetch_sources
python -m src.maltaekni_lokaverkefni.chunking
```

The chunking step also writes `data/processed/chunk_lemmas.json`. Lexical
methods such as TF-IDF and BM25 use that cache at startup. If the cache is
missing or was produced from older chunk text, the first run can spend a long
time lemmatizing every chunk again. Regenerate the cache after changing
chunking or searchable-text logic. Reynir/Greynir is used for proper Icelandic
lemmatization when installed; otherwise the app falls back to simple lowercase
tokens so local demos still start.

## LLM Answer Generation

The app can use Gemini or OpenAI to construct grounded answers from retrieved
source chunks. Gemini is the recommended provider for this project because
Gemini 3 performs strongly on Icelandic benchmarks and Flash is cost-oriented.

The local `.env` file follows `.env.example`:

```powershell
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3-flash-preview
LLM_MAX_OUTPUT_TOKENS=4096
```

`LLM_PROVIDER=auto` tries Gemini first and then OpenAI. If no provider key is set,
the app falls back to the local extractive answer generator so the demo still runs
without credentials.

The app records LLM usage metadata when the provider returns it. Evaluation rows
include input, output, thinking, and total token counts. Dollar estimates are only
calculated when optional per-million-token rates are set in `.env`, so the code
does not depend on hardcoded pricing.

## Evaluation

The evaluation script runs the fixed question set across retrieval methods
without using the UI:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25
```

The fixed evaluation plan is documented in `docs/evaluation_protocol.md`.
The `--no-llm` flag tests retrieval without spending LLM calls. Results are
written to `reports/evaluation/` as CSV and JSONL files.
The summary includes `expected_relevant_section` and
`expected_section_in_top_3` for the automatic retrieval check.
It also stores `confidence_reason`, which explains the answer confidence in
terms of source strength and citation coverage.
`source_coverage_ratio` records how many retrieved source chunks were actually
cited in the generated answer.
Each run also writes `evaluation_method_summary_latest.csv`, an aggregate table
by retrieval method for the report.

No-token smoke test:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --no-llm --limit 2
```

Model or prompt comparison runs can override settings from the command line:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --run-label gemini-strict --llm-provider gemini --gemini-model gemini-3-flash-preview --prompt-profile strict
```

Manual review UI:

```text
http://127.0.0.1:8000/evaluation
```

Report-table export:

```powershell
python -m src.maltaekni_lokaverkefni.export_report_tables
```

This writes CSV files to `reports/evaluation/report_tables/`:

- `table_retrieval_methods.csv`: top-3 retrieval score, source coverage, errors
- `table_cost_latency.csv`: latency, token counts, and estimated cost
- `table_human_scores.csv`: average 1-5 human scores by method
- `table_inter_reviewer.csv`: Sölvi/Jóhannes comparison
- `qualitative_cases.csv`: suggested examples for the error-analysis section

The main app also includes an `Aðferð` button with a short explanation of the
retrieval, Gemini answer generation, citations, and disclaimer.
Each source card includes a short reason for why that text was selected.

## External Review

The project is designed to run locally from the repository. For teacher review,
the Gemini key is provided privately as a temporary demo key and is never
committed to Git. The local `.env` file contains:

```text
GEMINI_API_KEY=...
```

After review, the temporary key is revoked. The repository keeps only
`.env.example`, which documents the required environment variables without
including secrets.
