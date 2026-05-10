# Máltækni lokaverkefni

Final project workspace for NLP / máltækni.

Author: Sölvi

## Status

Réttarvísir is a local Icelandic consumer-rights RAG prototype. It retrieves
legal source chunks, generates grounded Icelandic answers with citations, and
includes a no-token evaluation demo UI.

## Structure

- `src/maltaekni_lokaverkefni/`: reusable Python code
- `notebooks/`: exploratory notebooks
- `data/raw/`: local raw data, not committed
- `data/processed/`: local processed data, not committed
- `models/`: local model artifacts, not committed
- `reports/figures/`: generated figures, not committed by default
- `docs/`: project notes and non-private documentation

## Documentation

The main technical documentation is in `docs/codebase_documentation.md`. It
explains the full pipeline, module responsibilities, generated artifacts,
retrieval methods, answer generation, web endpoints, and evaluation workflow.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name maltaekni-lokaverkefni --display-name "Python (maltaekni-lokaverkefni)"
```

## Run Locally

Start the FastAPI web app:

```powershell
python -m uvicorn src.maltaekni_lokaverkefni.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

To open the chat directly without the welcome screen:

```text
http://127.0.0.1:8000/?skipWelcome=1
```

If port `8000` is already in use, choose another port:

```powershell
python -m uvicorn src.maltaekni_lokaverkefni.app:app --host 127.0.0.1 --port 8001
```

Then open:

```text
http://127.0.0.1:8001
```

The main app sidebar links to the evaluation dashboard and human review screens.
For development, the committed demo dataset can still be opened directly without
running evaluation or spending tokens:

```text
http://127.0.0.1:8000/evaluation?demo=1
http://127.0.0.1:8000/evaluation/dashboard?demo=1
```

To ask live questions through retrieval, make sure `data/processed/chunks.json`
exists. If it is missing, run:

```powershell
python -m src.maltaekni_lokaverkefni.fetch_sources
python -m src.maltaekni_lokaverkefni.chunking
```

## LLM answer generation

The app can use Gemini or OpenAI to construct grounded answers from retrieved
source chunks. Gemini is the recommended provider for this project because
Gemini 3 performs strongly on Icelandic benchmarks and Flash is cost-oriented.
Create a local `.env` file from `.env.example` and set:

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

Run the evaluation questions across retrieval methods without using the UI:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25
```

The fixed evaluation plan is documented in `docs/evaluation_protocol.md`.
Use `--no-llm` to test retrieval without spending LLM calls. Results are written
to `reports/evaluation/` as CSV and JSONL files.
The summary includes `expected_relevant_section` and
`expected_section_in_top_3` for the automatic retrieval check.
It also stores `confidence_reason`, which explains the answer confidence in
terms of source strength and citation coverage.
`source_coverage_ratio` records how many retrieved source chunks were actually
cited in the generated answer.
Each run also writes `evaluation_method_summary_latest.csv`, an aggregate table
by retrieval method for the report.

For a quick smoke test:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --no-llm --limit 2
```

For model or prompt comparisons, label each run and override the settings from
the command line instead of editing `.env`:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --run-label gemini-strict --llm-provider gemini --gemini-model gemini-3-flash-preview --prompt-profile strict
```

After running an evaluation, open the local review UI:

```text
http://127.0.0.1:8000/evaluation
```

The review UI saves human scores to one CSV per evaluator, for example
`reports/evaluation/evaluation_review_solvi.csv` and
`reports/evaluation/evaluation_review_johannes.csv`. These review CSVs are
intended to be committed, so Sölvi and Jóhannes can review on separate machines
and then pull each other's files before writing the report.
The dashboard at `http://127.0.0.1:8000/evaluation/dashboard` summarizes the
latest automatic metrics, token usage, and all committed human review files
without making new LLM calls.
After finishing a review session, commit only your evaluator file:

```powershell
git add reports/evaluation/evaluation_review_solvi.csv
git commit -m "Add Solvi human evaluation reviews"
git pull --rebase origin main
git push origin main
```

Jóhannes uses `reports/evaluation/evaluation_review_johannes.csv` in the same
workflow.

To export report-ready tables after the final automatic run and human review:

```powershell
python -m src.maltaekni_lokaverkefni.export_report_tables
```

This writes CSV files to `reports/evaluation/report_tables/`:

- `table_retrieval_methods.csv`: top-3 retrieval score, source coverage, errors
- `table_cost_latency.csv`: latency, token counts, and estimated cost
- `table_human_scores.csv`: average 1-5 human scores by method
- `table_inter_reviewer.csv`: Sölvi/Jóhannes comparison
- `qualitative_cases.csv`: suggested examples for the error-analysis section

To show the evaluation UI without running evaluation first, use the committed
demo dataset:

```text
http://127.0.0.1:8000/evaluation?demo=1
http://127.0.0.1:8000/evaluation/dashboard?demo=1
```

The main app also includes an `Aðferð` button with a short explanation of the
retrieval, Gemini answer generation, citations, and disclaimer.
Each source card includes a short reason for why that text was selected.

## Notes

The local PDF handouts are intentionally ignored so the public GitHub repository starts clean.
