"""Shared data shapes passed between pipeline stages.

`Document` is the cleaned page-level source format, `Chunk` is the article-level
retrieval format, and `EvaluationRow` is the flattened CSV schema used for
automatic evaluation and report tables.
"""


from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Document:
    """One cleaned source document before article-based chunking."""
    document_id: str
    text: str
    source: str
    title: str
    url: str
    

@dataclass(frozen=True)
class Chunk:
    """One searchable source chunk returned by retrievers and cited in answers."""
    chunk_id: str
    text: str
    source: str
    title: str
    url: str
    section: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the chunk for JSON artifacts and API responses."""
        return asdict(self)
    
@dataclass(frozen=True)
class EvaluationRow:
    """One reportable result for a question, retrieval method, and answer run."""
    run_label: str
    question_id: str
    question: str
    topic: str
    case_type: str
    expected_behavior: str
    expected_relevant_section: str
    retrieval_method: str
    answer_method: str
    top_1_chunk_id: str
    top_1_title: str
    top_1_section: str
    top_1_score: float | None
    top_3_sections: str
    retrieval_check_applicable: bool
    expected_section_in_top_3: bool
    confidence: str
    confidence_reason: str
    prompt_profile: str
    cited_source_count: int | None
    source_count: int | None
    source_coverage_ratio: float | None
    llm_provider: str
    llm_model: str
    prompt_tokens: int | None
    output_tokens: int | None
    thought_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None
    latency_seconds: float
    answer: str
    source_urls: str
    error: str = ""
