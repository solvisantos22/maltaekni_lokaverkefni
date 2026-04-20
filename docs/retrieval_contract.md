# Retrieval Contract

This document defines the format that the retrieval component should return to the answer-generation and UI components.

Owner of downstream use: Sölvi

## Goal

The answer-generation component should not need to know how retrieval works. It only needs a user question and a ranked list of source chunks.

## Python Type

```python
retrieval_result = {
    "question": "Hvað get ég gert ef vara er gölluð?",
    "chunks": [
        {
            "chunk_id": "neytendastofa_gallar_001",
            "text": "Relevant source text...",
            "source": "Neytendastofa",
            "title": "Gölluð vara",
            "url": "https://example.com/source",
            "section": "Kvartanir vegna galla",
            "published_or_updated": "2026",
            "score": 0.82
        }
    ]
}
```

## Required Fields

Each chunk must include:

| Field | Type | Purpose |
|---|---|---|
| `chunk_id` | string | Stable ID for debugging and evaluation |
| `text` | string | Source text used by the answer generator |
| `source` | string | Source name, for example `Neytendastofa` or `Althingi` |
| `title` | string | Human-readable source title |
| `url` | string | Link to the original source |
| `score` | number | Retrieval score, where higher means more relevant |

## Optional But Useful Fields

| Field | Type | Purpose |
|---|---|---|
| `section` | string | Law article, heading, or page section |
| `published_or_updated` | string | Publication/update year or date if known |
| `retrieval_method` | string | For example `bm25`, `tfidf`, `embedding`, or `hybrid` |

## Ranking

The chunks should be sorted from most relevant to least relevant. The answer-generation component will usually use the top 1-3 chunks.

## Text Length

Recommended chunk length:

- Minimum: about 80 words
- Target: about 150-250 words
- Maximum: about 350 words

Chunks that are too short may lack context. Chunks that are too long make citation and grounding harder.

## Empty Results

If no relevant source is found, return an empty chunk list:

```python
{
    "question": "Some question",
    "chunks": []
}
```

The answer component should then say that it does not have enough source information to answer reliably.
