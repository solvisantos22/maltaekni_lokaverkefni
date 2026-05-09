# Evaluation Protocol

This document defines the final evaluation setup before running paid LLM tests.
The goal is to avoid changing prompts, models, or metrics after spending tokens.

## Dataset

Evaluation questions live in `docs/evaluation_questions.csv`.

The current set has 30 questions:

- direct consumer-rights questions
- paraphrased questions that should still retrieve the same legal section
- ambiguous or multi-source questions
- advertising and marketing-law questions
- no-source questions where the system should avoid unsupported advice

Rows with `expected_relevant_section=NO_RELEVANT_SOURCE` are used for human
uncertainty evaluation. They are not included in the automatic top-3 legal
section score.

## Fixed Test Matrix

Primary paid evaluation:

| Setting | Value |
|---|---|
| Retrieval methods | `tfidf`, `bm25`, `bge-m3`, `rrf-bge-m3-bm25` |
| LLM provider | `gemini` |
| LLM model | `gemini-3-flash-preview` |
| Prompt profile | `strict` |
| Top-k retrieved chunks | `3` |
| Questions | all rows in `docs/evaluation_questions.csv` |
| Run label | `gemini-strict-final` |

Optional prompt comparison, only if token budget allows:

| Setting | Value |
|---|---|
| Retrieval methods | `tfidf`, `bm25` |
| LLM provider | `gemini` |
| LLM model | `gemini-3-flash-preview` |
| Prompt profiles | `balanced`, `strict` |
| Top-k retrieved chunks | `3` |
| Questions | first 10 representative questions |
| Run labels | `gemini-balanced-sample`, `gemini-strict-sample` |

Do not rerun the full paid matrix unless the code, prompt, or data changes in a
way that must be reflected in the report.

## Commands

No-token smoke test:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --no-llm --limit 4 --run-label smoke-no-llm
```

Primary paid run:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 bge-m3 rrf-bge-m3-bm25 --llm-provider gemini --gemini-model gemini-3-flash-preview --prompt-profile strict --run-label gemini-strict-final
```

Optional prompt comparison:

```powershell
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --limit 10 --llm-provider gemini --gemini-model gemini-3-flash-preview --prompt-profile balanced --run-label gemini-balanced-sample
python -m src.maltaekni_lokaverkefni.evaluate_methods --methods tfidf bm25 --limit 10 --llm-provider gemini --gemini-model gemini-3-flash-preview --prompt-profile strict --run-label gemini-strict-sample
```

## Automatic Metrics

The evaluation script saves one row per question and retrieval method.

Report these automatic metrics:

- `expected_section_in_top_3`: whether the expected legal section appears in the top 3 retrieved chunks.
- `expected_section_top3_rate`: aggregate hit rate over rows where a legal section is expected.
- `source_coverage_ratio`: share of retrieved source chunks that the answer actually cites.
- `latency_seconds`: total retrieval and answer-generation time.
- `prompt_tokens`, `output_tokens`, `thought_tokens`, `total_tokens`: provider-reported usage when available.
- `estimated_cost_usd`: only used if local per-million-token rates are configured in `.env`.
- `error`: whether the method failed on a question.

The script also writes `reports/evaluation/evaluation_method_summary_latest.csv`
for report-ready aggregate tables.

After the final run and human review, export the report tables with:

```powershell
python -m src.maltaekni_lokaverkefni.export_report_tables
```

The exporter writes clean CSV tables to `reports/evaluation/report_tables/`.
These tables are intended for direct use in the written report:

- retrieval-method comparison
- latency, token, and cost comparison
- human quality scores
- inter-reviewer comparison
- qualitative case candidates for error analysis

## Human Evaluation

After a run, open:

```text
http://127.0.0.1:8000/evaluation
```

Sölvi and Jóhannes should each score all rows, or at minimum the final selected
method on all questions plus a comparison sample from the other methods.

Human scores use a 1-5 scale:

| Field | Meaning |
|---|---|
| `retrieval_relevance_1_5` | Were the retrieved sources relevant to the question? |
| `answer_correctness_1_5` | Was the answer correct according to the shown sources? |
| `source_support_1_5` | Were the answer claims supported by citations? |
| `clarity_1_5` | Was the Icelandic clear and useful for a consumer? |

Use the notes field for recurring failure modes:

- wrong legal article
- right article but incomplete answer
- answer too broad or too legalistic
- unsupported claim
- missed uncertainty
- confusing Icelandic wording

## Dashboard

After running evaluation, open:

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

The final product does not need to be perfect. The report should explain which
methods worked better, where the system failed, and why those failures are
reasonable for the current data and model setup.
