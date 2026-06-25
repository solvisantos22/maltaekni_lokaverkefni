# Réttarvísir

Réttarvísir is a local Icelandic consumer-rights RAG prototype built for the
University of Iceland course TÖL025M, Introduction to NLP.

The system answers Icelandic questions about consumer law by retrieving relevant
legal source chunks, generating a grounded answer, and showing the exact
citations used. It also includes an evaluation dashboard and a manual review UI
used to compare retrieval methods.

Authors: Sölvi Santos and Jóhannes Reykdal Einarsson.

## What This Project Demonstrates

- Retrieval-augmented generation for a low-resource language domain.
- Icelandic legal text processing, including tokenization and lemmatization.
- Retrieval comparison across TF-IDF, BM25, embedding search, reciprocal rank
  fusion, and reranking.
- Source-grounded answer generation with Gemini/OpenAI support and a local
  extractive fallback.
- Human evaluation workflow for checking answer quality, source support, and
  clarity.
- Report-ready exports for tables, cost/latency, and qualitative examples.

The goal was not to build production legal advice software. The project is an
educational prototype for understanding how retrieval choices, legal language,
Icelandic morphology, and source grounding affect answer quality.

## Demo Flow

For a quick local demo:

1. Open the chat UI.
2. Ask: `Hvaða úrræði hefur neytandi ef söluhlutur reynist gallaður?`
3. Read the answer citations and source cards.
4. Switch between `BM25`, `TF-IDF`, and one RRF method.
5. Open the evaluation dashboard at `/evaluation/dashboard`.
6. Open the manual review UI at `/evaluation`.

The UI is in Icelandic because the domain, data, and evaluation questions are in
Icelandic.

## Repository Structure

- `src/maltaekni_lokaverkefni/`: backend, retrieval, answer generation,
  evaluation, and report-export code.
- `src/maltaekni_lokaverkefni/web/`: HTML, CSS, and JavaScript for the chat,
  teacher guide, evaluation dashboard, and review screens.
- `docs/`: technical documentation, methodology notes, evaluation protocol, and
  evaluation questions.
- `reports/evaluation/`: saved evaluation outputs, human review CSVs, and
  exported report tables.
- `reports/final_report.pdf`: final Icelandic project report.
- `data/raw/`: raw supporting data such as Icelandic stop words.
- `data/processed/`: generated local data artifacts. These are intentionally not
  committed by default.

## Quick Start

Python 3.11 or 3.12 is recommended.

### 1. Create an Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Generate Local Data

The live app needs processed legal chunks:

```powershell
python -m src.maltaekni_lokaverkefni.fetch_sources
python -m src.maltaekni_lokaverkefni.chunking
```

The second command also writes `data/processed/chunk_lemmas.json`. TF-IDF and
BM25 use that cache at startup. If the cache is missing or stale, startup can be
slow because the app has to analyze every chunk again.

### 3. Configure Optional LLM Access

The app runs without an API key by using a local extractive fallback:

```text
LLM_PROVIDER=none
```

For grounded generated answers, create a local `.env` file based on
`.env.example`:

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3-flash-preview
LLM_MAX_OUTPUT_TOKENS=4096
```

Secrets are never committed to Git.

### 4. Run the App

```powershell
python -m uvicorn src.maltaekni_lokaverkefni.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/?skipWelcome=1
```

Useful local URLs:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/evaluation/dashboard
http://127.0.0.1:8000/evaluation
```

## Architecture

The pipeline has seven main stages:

1. `fetch_sources.py` downloads curated Althingi law pages.
2. `chunking.py` splits laws into article-level chunks and writes the lemma
   cache.
3. `retriever.py` ranks chunks for a question using the selected retrieval
   method.
4. `prompts.py` builds a grounded answer prompt from the top chunks.
5. `answer_generator.py` calls Gemini/OpenAI when configured, or uses the local
   fallback.
6. `app.py` exposes the FastAPI routes and static web UI.
7. `evaluate_methods.py` and `export_report_tables.py` produce evaluation and
   report artifacts.

The lexical retrievers and the cache writer share the same searchable-text
function. This matters: if the cached text and runtime text differ, the SHA256
hashes do not match and the app has to reprocess all chunks.

## Retrieval Methods

Supported methods:

- `tfidf`: lexical TF-IDF over analyzed Icelandic tokens.
- `bm25`: BM25 over analyzed Icelandic tokens.
- `icebert`: direct embedding retrieval with IceBERT.
- `bge-m3`: direct embedding retrieval with BGE-M3.
- `rrf-icebert-bm25`: reciprocal rank fusion between BM25 and IceBERT.
- `rrf-bge-m3-bm25`: reciprocal rank fusion between BM25 and BGE-M3.
- `rrf-bge-m3-bm25-rerank`: BGE-M3/BM25 fusion followed by BGE reranking.

The embedding and reranking methods download Hugging Face models on first use
and can be heavy on a normal laptop. For a fast demo, start with `BM25` or
`TF-IDF`.

## Evaluation

The fixed evaluation dataset is in `docs/evaluation_questions.csv`, and the
protocol is documented in `docs/evaluation_protocol.md`.

Run a no-token smoke test:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --no-llm --limit 2
```

Run selected methods:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 rrf-bge-m3-bm25
```

Export report tables:

```powershell
python -m src.maltaekni_lokaverkefni.export_report_tables
```

The dashboard reads saved evaluation artifacts only. It does not start new LLM
calls.

## Documentation

Start here:

- `docs/codebase_documentation.md`: full technical overview.
- `docs/evaluation_protocol.md`: evaluation setup and scoring criteria.
- `docs/methodology_summary.md`: short methodology explanation.
- `reports/final_report.pdf`: final report in Icelandic.

The codebase also includes module docstrings and inline comments for the main
pipeline decisions.

## Known Limitations

- The app is not legal advice.
- The source corpus is intentionally limited to selected consumer-rights legal
  sources.
- Icelandic legal terms often differ from everyday language, which is one of the
  core limitations studied in the project.
- Neural retrieval and reranking can be slow or memory-heavy on first use because
  model weights need to be downloaded and loaded.
- If `Reynir` is unavailable, the tokenizer falls back to simpler regex tokens
  so the app can still run locally, but proper Icelandic lemmatization is the
  intended setup.
