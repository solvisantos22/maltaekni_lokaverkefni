# Methodology Summary

This is report-ready methodology text for Réttarvísir. It is written as source
material for the final report, not as UI copy.

## Project Goal

Réttarvísir is an Icelandic consumer-rights question-answering prototype. The
system answers user questions in Icelandic, retrieves legal source material, and
shows citations so the user can inspect the evidence behind each answer. The
goal is not to replace legal advice, but to test how well Icelandic NLP methods
can support grounded retrieval and answer generation for a narrow legal domain.

## Data Sources

The corpus is built from curated Icelandic legal and consumer-rights sources.
The current core source is Althingi/Lagasafn legal text, especially laws related
to consumer purchases, contracts, marketing, and cancellation rights. Source
fetching downloads HTML pages, removes navigation and amendment noise, and saves
clean documents with title, URL, source name, document ID, and text.

## Chunking

Legal texts are chunked by article where possible, using headings such as
`26. gr.` as natural boundaries. This preserves legal context better than fixed
character windows because consumer-rights answers often depend on one specific
article. Each chunk stores searchable text plus metadata such as title, section,
URL, source, and chunk ID.

## Retrieval Methods

The project compares several retrieval methods:

- TF-IDF lexical retrieval
- BM25 lexical retrieval
- embedding retrieval with IceBERT/BGE-M3 style models
- reciprocal rank fusion between lexical and embedding retrieval
- reranked retrieval where a cross-encoder scores candidate chunks after the
  first-stage retriever

The lexical methods use Icelandic tokenization and lemmatization where available.
The embedding and reranking methods test whether neural representations improve
source ranking for paraphrased or less literal questions.

## Answer Generation

After retrieval, the system sends the user question and selected chunks to an
LLM answer generator. Gemini 3 Flash is used as the main answer model because it
is relatively cheap and strong enough for Icelandic answer generation. The
prompt instructs the model to answer only from the retrieved sources, write in
clear Icelandic, and cite source IDs next to supported claims. If no LLM key is
available, the app falls back to an extractive answer generator so the local app
can still run without API calls.

## Citation Grounding

Each retrieved source is assigned a citation ID. The generated answer is checked
against the retrieved source list, and the UI displays both the answer and the
underlying source snippets. The evaluation also records source coverage: the
share of retrieved chunks that were actually cited by the answer.

## Evaluation Design

The automatic evaluation uses a fixed CSV of Icelandic consumer-rights
questions. Each question includes expected behavior and, when applicable, an
expected legal section. Retrieval quality is measured by whether the expected
section appears among the top three retrieved chunks. The pipeline also records
latency, errors, cited source count, source coverage, token usage, and estimated
cost when the provider returns usage metadata.

Human evaluation is done by Sölvi and Jóhannes in a separate review UI. Each
reviewer scores the same answers on four 1-5 dimensions: retrieval relevance,
answer correctness, source support, and clarity. Scores are saved to separate
CSV files per reviewer and combined by the dashboard/export script.

## Report Outputs

The report tables are generated from saved evaluation artifacts, not from live
LLM calls. The exporter reads the latest automatic evaluation CSV/JSONL files and
the human review CSVs, then writes report-ready tables for retrieval quality,
token/cost/latency, human scores, reviewer agreement, and qualitative examples.

## Limitations

The corpus is narrow and does not cover all Icelandic consumer-law resources.
Automatic top-3 scoring only checks expected legal sections and does not fully
measure legal correctness. Human evaluation is therefore necessary for answer
quality, citation support, and clarity. The system should be presented as an
information tool, not as legal advice.
