"""
Defines classes for our documents and chunks
"""


from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Document:
    """One cleaned source document before chunking."""
    document_id: str
    text: str
    source: str
    title: str
    url: str
    

@dataclass(frozen=True)
class Chunk:
    """One searchable chunk matching the retrieval contract metadata."""
    chunk_id: str
    text: str
    source: str
    title: str
    url: str
    section: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)