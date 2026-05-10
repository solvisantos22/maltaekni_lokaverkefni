# Evaluation Protocol

This document describes the evaluation setup used for the final experiments.
The protocol fixes the dataset, retrieval methods, answer model, metrics, and
human review criteria so the reported results are reproducible.

## Dataset

Evaluation questions live in `docs/evaluation_questions.csv`.

The final set has 20 questions. The set is intentionally smaller than the
earlier 30-question version because the final matrix includes several retrieval
methods, and every method-question pair also needs human review.

The core design is paired evaluation:

- 8 questions use legal terminology that appears in the source laws, for example
  `söluhlutur`, `úrræði`, `afhendingardráttur`, and `fjarsölusamningur`.
- 8 questions ask for the same expected legal sections in everyday consumer
  language, for example `varan`, `búðin`, `pöntunin`, and `hætta við netkaup`.
- 2 questions are ambiguous or multi-source questions.
- 2 questions are no-source questions where the system should avoid unsupported
  advice.

The paired legal/everyday wording is part of the experiment. It tests whether
retrieval methods depend on matching legal vocabulary exactly, and whether
embedding, fusion, or reranking methods degrade less when users phrase questions
in ordinary language.

Rows with `expected_relevant_section=NO_RELEVANT_SOURCE` are used for human
uncertainty evaluation. They are not included in the automatic top-3 legal
section score.

## Fixed Test Matrix

Final paid evaluation:

| Setting | Value |
|---|---|
| Retrieval methods | `tfidf`, `bm25`, `rrf-icebert-bm25`, `rrf-bge-m3-bm25`, `rrf-bge-m3-bm25-rerank` |
| LLM provider | `gemini` |
| LLM model | `gemini-3-flash-preview` |
| Prompt profile | `strict` |
| Top-k retrieved chunks | `3` |
| Questions | all rows in `docs/evaluation_questions.csv` |
| Run label | `gemini-strict-final` |

Optional prompt comparison design:

| Setting | Value |
|---|---|
| Retrieval methods | `tfidf`, `bm25` |
| LLM provider | `gemini` |
| LLM model | `gemini-3-flash-preview` |
| Prompt profiles | `balanced`, `strict` |
| Top-k retrieved chunks | `3` |
| Questions | first 10 representative questions |
| Run labels | `gemini-balanced-sample`, `gemini-strict-sample` |

The full paid matrix is treated as a saved experimental artifact. Re-running it
changes the report inputs and may incur additional token cost.

## Commands

No-token smoke test:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --no-llm --limit 4 --run-label smoke-no-llm
```

Final paid run:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 rrf-icebert-bm25 rrf-bge-m3-bm25 rrf-bge-m3-bm25-rerank --llm-provider gemini --gemini-model gemini-3-flash-preview --prompt-profile strict --run-label gemini-strict-final
```

Optional prompt comparison:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --limit 10 --llm-provider gemini --gemini-model gemini-3-flash-preview --prompt-profile balanced --run-label gemini-balanced-sample
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --limit 10 --llm-provider gemini --gemini-model gemini-3-flash-preview --prompt-profile strict --run-label gemini-strict-sample
```

## Automatic Metrics

The evaluation script saves one row per question and retrieval method. The
automatic metrics recorded for the report are:

- `expected_section_in_top_3`: whether the expected legal section appears in the top 3 retrieved chunks.
- `expected_section_top3_rate`: aggregate hit rate over rows where a legal section is expected.
- `source_coverage_ratio`: share of retrieved source chunks that the answer actually cites.
- `latency_seconds`: total retrieval and answer-generation time.
- `prompt_tokens`, `output_tokens`, `thought_tokens`, `total_tokens`: provider-reported usage when available.
- `estimated_cost_usd`: only used if local per-million-token rates are configured in `.env`.
- `error`: whether the method failed on a question.

The script also writes `reports/evaluation/evaluation_method_summary_latest.csv`
for report-ready aggregate tables.

The final run and human review artifacts are converted to report tables with:

```powershell
python -m src.maltaekni_lokaverkefni.export_report_tables
```

The exporter writes clean CSV tables to `reports/evaluation/report_tables/`:

- retrieval-method comparison
- latency, token, and cost comparison
- human quality scores
- inter-reviewer comparison
- qualitative case candidates for error analysis

## Human Evaluation

Manual review page:

```text
http://127.0.0.1:8000/evaluation
```

Sölvi and Jóhannes score the same saved answer rows so the report can compare
automatic metrics with independent human judgments.

Human scores use a 1-5 scale:

| Field | Meaning |
|---|---|
| `retrieval_relevance_1_5` | Were the retrieved sources relevant to the question? |
| `answer_correctness_1_5` | Was the answer correct according to the shown sources? |
| `source_support_1_5` | Were the answer claims supported by citations? |
| `clarity_1_5` | Was the Icelandic clear and useful for a consumer? |

The notes field captures recurring failure modes:

- wrong legal article
- right article but incomplete answer
- answer too broad or too legalistic
- unsupported claim
- missed uncertainty
- confusing Icelandic wording

## Dashboard

Dashboard page:

```text
http://127.0.0.1:8000/evaluation/dashboard
```

The dashboard reads saved CSV/JSONL files only. It does not make LLM calls and
does not spend tokens.

## Reporting Plan

In the report, present results in this order:

1. Automatic retrieval quality by method.
2. Token usage and latency by method.
3. Human scores for answer quality.
4. Error analysis with concrete examples.
5. Final method choice and limitations.

The report explains which methods worked better, where the system failed, and
why those failures are reasonable for the final data and model setup.
