# Data

The project separates original source material from generated pipeline
artifacts.

- `data/raw/`: local raw source files, if any are downloaded separately.
- `data/processed/`: generated documents, chunks, lemma caches, and embedding
  caches used by retrieval.

Most data artifacts are ignored by Git because they can be regenerated from the
source-fetching and chunking scripts.
