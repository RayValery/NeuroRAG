from dataclasses import dataclass

@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    journal: str |None
    has_fulltext: bool = False


@dataclass
class Chunk:
    pmid: str
    section: str
    chunk_index: int
    content: str
    embedding: list[float] | None = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    paper_title: str
    year: int | None
    score: float