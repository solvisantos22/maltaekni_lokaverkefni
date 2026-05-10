"""Template helpers for turning parsed documents into retrieval chunks."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

try:
    from .ice_tokenizer import IceTokenizer
    from .types_classes import Document, Chunk
except ImportError:  # Allows direct script execution during early experiments.
    from ice_tokenizer import IceTokenizer
    from types_classes import Document, Chunk


ARTICLE_RE = re.compile(r"^\[?\d+\. gr\.?.*")
LEMMA_CACHE_VERSION = 1


class Chunker():
    """
    Convert parsed documents into retrieval chunks.

    Explanation:
        Chunker takes cleaned Document objects and splits them into smaller
        Chunk objects that can be indexed by the retriever. For Althingi laws,
        the current strategy is one chunk per legal article, using lines such as
        "26. gr. Úrræði neytanda vegna galla." as section boundaries. This
        preserves citation-friendly legal sections while keeping retrieval
        inputs small enough for ranking and answer generation.

    Attributes:
        documents: Documents waiting to be chunked.
        chunks: Flat list of Chunk objects produced from the documents.

    Public methods:
        chunk_documents(): Chunk every document stored in documents.
        get_chunks(): Return the generated Chunk objects.
    """
    
    documents: list[Document] | list = []
    chunks: list[Chunk] | list = []

    def __init__(self, documents: list[Document] | None = None):
        """Initialize the chunker with an optional list of documents."""
        self.documents = documents or []
        self.chunks = []
    

    def chunk_documents(self):
        """Chunk every loaded document using the strategy for its source."""
        for document in self.documents:
            self.__chunk_document(document)
            
    def get_chunks(self):
        """Return generated chunks."""
        return self.chunks

    def lemmatize_chunks(self, path: Path):
        """Lemmatize searchable chunk text and write a retrieval-token cache."""
        tokenizer = IceTokenizer()
        cache = {
            "version": LEMMA_CACHE_VERSION,
            "tokenizer": "IceTokenizer.lemmatIce",
            "chunks": {},
        }

        for chunk in self.chunks:
            searchable_text = searchable_text_for_chunk(chunk)
            cache["chunks"][chunk.chunk_id] = {
                "text_hash": text_hash(searchable_text),
                "lemmas": tokenizer.lemmatIce(searchable_text),
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(cache, file, ensure_ascii=False, indent=2)

        return cache

    def __chunk_document(self, document: Document):
        """Choose the chunking strategy for one document."""
        if document.source.lower() == 'althingi':
            self.__chunk_althingi_law(document)
        else:
            raise ValueError("The source must be althingi")
        

    def __chunk_althingi_law(self, document: Document):
        """Split an Althingi law into one chunk per legal article."""
        current_section: str | None = None
        current_lines: list[str] = []
        chunk_number = 1

        for line in self.__clean_lines(document.text):
            if ARTICLE_RE.match(line):
                if current_section and current_lines:
                    self.chunks.append(self.__make_chunk(document, current_section, current_lines, chunk_number))
                    chunk_number += 1

                current_section = line
                current_lines = [line]
            elif current_section:
                current_lines.append(line)

        if current_section and current_lines:
            self.chunks.append(self.__make_chunk(document, current_section, current_lines, chunk_number))



    def __make_chunk(self, document: Document, section: str, lines: list[str], number: int) -> Chunk:
        """Create one Chunk object from collected article lines."""
        return Chunk(
            chunk_id=f"{document.document_id}_{number:03d}",
            text="\n".join(lines),
            source=document.source,
            title=document.title,
            url=document.url,
            section=section,
        )


    def __clean_lines(self, text: str) -> list[str]:
        """Normalize whitespace and remove empty lines."""
        return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def load_documents(path: Path) -> list[Document]:
    """Load Document objects from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        raw_documents = json.load(file)

    return [Document(**raw_document) for raw_document in raw_documents]


def save_chunks(chunks: list[Chunk], path: Path):
    """Save Chunk objects to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump([asdict(chunk) for chunk in chunks], file, ensure_ascii=False, indent=2)


def searchable_text_for_chunk(chunk: Chunk) -> str:
    """Build indexed text with repeated section headings for retrieval weight."""
    return "\n".join(
        [
            chunk.title,
            chunk.section,
            chunk.section,
            chunk.section,
            chunk.text,
        ]
    )


def text_hash(text: str) -> str:
    """Return a stable hash for cached searchable text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    input_path = Path("data/processed/documents.json")
    output_path = Path("data/processed/chunks.json")

    documents = load_documents(input_path)
    chunker = Chunker(documents)
    chunker.chunk_documents()
    chunks = chunker.get_chunks()
    save_chunks(chunks, output_path)
    chunker.lemmatize_chunks(output_path.parent / "chunk_lemmas.json")

    print(f"Saved {len(chunks)} chunks to {output_path}")
