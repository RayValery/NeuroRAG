# Neuro-Literature RAG — Semantic Search & Grounded Answering

**Status:** Draft
**Created:** 2026-08-09

---

## Executive Summary

This is the first component of the NeuroRAG platform: a pipeline that ingests neuroscience papers from PubMed/PMC, indexes them for semantic (meaning-based) retrieval in a pgvector store, and uses an LLM to produce grounded, cited answers to user questions. It is deliberately not a keyword search — it wins over PubMed on *conceptual* retrieval and *cross-paper synthesis*, and it is the foundation the later analysis-tool and agent layers build on.

---

## Goals

- Pull neuroscience papers (abstracts first, open-access full text later) from PubMed/PMC via the Entrez API.
- Index paper passages as embeddings in pgvector with full citation metadata.
- Answer natural-language questions by retrieving semantically relevant passages and having an LLM synthesize a cited answer strictly grounded in them.
- Keep the LLM behind a swappable interface so a hosted API can later be replaced by a self-hosted model with a one-file change.

## Non-Goals

- No agent / tool-calling orchestration yet (later phase).
- No served neuro ML model (EEG/spike decoder) yet (later phase).
- No GraphRAG or knowledge graph yet (later phase).
- Not trying to beat PubMed at keyword lookup — synthesis and semantic recall are the point.

---

## Background & Context

PubMed is a keyword/metadata retrieval engine: it returns documents, and the user does the reading and synthesis. This system adds a semantic-retrieval-plus-synthesis layer on top of PubMed (which serves as the *ingestion source*, not a competitor). Semantic search surfaces conceptually related work that shares no keywords; the LLM step turns a set of passages into a single grounded answer with citations. This is also the seam the later neuro-model work depends on — contextualizing a model's numeric output against the literature is something keyword search structurally cannot do.

---

## Requirements

### Must Have
- Entrez `esearch` + `efetch`/`esummary` ingestion of PMIDs → title, abstract, authors, year, journal, PMID.
- Structure-aware chunking with per-chunk metadata (paper ID, title, section, year).
- Local embedding via a biomedical sentence-transformer.
- pgvector store with cosine similarity search and metadata filtering (e.g. year range).
- Retrieval → LLM answer that cites which paper each claim came from and answers only from retrieved passages.
- LLM access behind an abstract interface (provider-swappable).
- Graceful rate-limit handling for Entrez (respect 3 req/s default, 10 with API key).

### Should Have
- Open-access full-text pull from PMC (XML) with fallback to abstract-only.
- Deduplication by PMID on ingest (idempotent re-runs).
- A "low retrieval confidence → refuse / say unsure" guardrail.
- Configurable top-k and similarity threshold.

### Nice to Have
- Hybrid search (BM25 + vector) as a later toggle.
- Simple CLI or minimal FastAPI endpoint for querying.
- Caching of embeddings to avoid recompute.

---

## Architecture Overview

```
                   ┌─────────────────────────────────────────────┐
   PubMed/PMC ───► │ Ingestion (Entrez client)                   │
   (Entrez API)    │  esearch → PMIDs → efetch → metadata/text   │
                   └──────────────────┬──────────────────────────┘
                                      ▼
                   ┌─────────────────────────────────────────────┐
                   │ Chunker (structure-aware, +metadata)        │
                   └──────────────────┬──────────────────────────┘
                                      ▼
                   ┌─────────────────────────────────────────────┐
                   │ Embedder (biomedical sentence-transformer)  │
                   └──────────────────┬──────────────────────────┘
                                      ▼
                   ┌─────────────────────────────────────────────┐
                   │ pgvector store  (papers, chunks + vectors)  │
                   └──────────────────┬──────────────────────────┘
                                      ▼
   user query ───► Embed query ─► Vector search (top-k, filters) ─┐
                                                                  ▼
                   ┌─────────────────────────────────────────────┐
                   │ LLM answerer (behind LLMClient interface)   │
                   │  prompt = query + retrieved passages        │
                   │  output = grounded, cited answer            │
                   └─────────────────────────────────────────────┘
```

**Components**
- **EntrezClient** — search + fetch, rate-limited, returns normalized `Paper` records.
- **Chunker** — splits abstract/sections into ~300–500 token chunks with overlap, carries metadata.
- **Embedder** — wraps a sentence-transformer; one method for docs, one for queries.
- **VectorStore** — pgvector-backed; upsert chunks, similarity search with filters.
- **LLMClient** (interface) — `HostedLLMClient` now; `SelfHostedLLMClient` (vLLM/Ollama) later.
- **Answerer** — orchestrates embed-query → retrieve → build prompt → call LLMClient.

---

## Data Models

```sql
-- Papers: one row per PMID
CREATE TABLE papers (
    pmid        TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    abstract    TEXT,
    authors     TEXT[],          -- ordered author list
    year        INT,
    journal     TEXT,
    has_fulltext BOOLEAN DEFAULT FALSE,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

-- Chunks: many per paper, each embedded
CREATE TABLE chunks (
    id          BIGSERIAL PRIMARY KEY,
    pmid        TEXT REFERENCES papers(pmid) ON DELETE CASCADE,
    section     TEXT,            -- 'abstract' | 'methods' | 'results' | ...
    chunk_index INT,             -- order within the paper
    content     TEXT NOT NULL,
    embedding   VECTOR(768)      -- dim must match the embedding model
);

-- ANN index for cosine similarity
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON papers (year);
```

```python
# Python domain types (dataclasses)
@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    journal: str | None
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
    score: float          # cosine similarity
```

> Set `VECTOR(n)` to your model's output dim. `S-PubMedBert-MS-MARCO` and `bge-base` are 768; `bge-large` is 1024.

---

## API / Interface Definitions

```python
class EntrezClient:
    def search(self, term: str, retmax: int = 100,
               mindate: int | None = None, maxdate: int | None = None) -> list[str]:
        """esearch → list of PMIDs. Rate-limited."""

    def fetch(self, pmids: list[str]) -> list[Paper]:
        """efetch/esummary → normalized Paper records (batched)."""

class Chunker:
    def chunk(self, paper: Paper) -> list[Chunk]:
        """Structure-aware split into ~300-500 token chunks with overlap + metadata."""

class Embedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class VectorStore:
    def upsert_paper(self, paper: Paper) -> None: ...
    def upsert_chunks(self, chunks: list[Chunk]) -> None: ...
    def search(self, query_vec: list[float], top_k: int = 8,
               min_year: int | None = None) -> list[RetrievedChunk]: ...

class LLMClient(Protocol):                      # swappable boundary
    def complete(self, system: str, user: str) -> str: ...

class Answerer:
    def answer(self, question: str, top_k: int = 8,
               min_score: float = 0.35) -> Answer:
        """embed → retrieve → (guardrail) → prompt → LLM → cited answer."""

@dataclass
class Answer:
    text: str
    citations: list[dict]     # [{pmid, title, year, score}]
    used_chunks: int
    refused: bool = False     # true if retrieval confidence too low
```

**Grounding prompt contract (system message, essentials):**
- Answer *only* from the provided passages; if they don't contain the answer, say so.
- Every factual claim must reference the paper it came from (title/PMID).
- Do not invent citations or facts not present in the passages.

---

## Implementation Plan

### Phase 1: Infrastructure
- [ ] `docker-compose` with Postgres + pgvector; run the schema DDL.
- [ ] Config (env): Entrez email/API key, DB URL, embedding model name, LLM provider/key.
- [ ] Domain dataclasses.

### Phase 2: Ingestion
- [ ] `EntrezClient.search` (esearch) with date filters + rate limiting.
- [ ] `EntrezClient.fetch` (efetch/esummary), batched, parse into `Paper`.
- [ ] Idempotent upsert of papers (dedupe by PMID).
- [ ] (Should) PMC full-text XML fetch with abstract-only fallback.

### Phase 3: Indexing
- [ ] `Chunker` — abstract-first, structure-aware, overlap, metadata.
- [ ] `Embedder` wrapping the biomedical model (doc vs query methods).
- [ ] `VectorStore.upsert_chunks` writing vectors to pgvector.
- [ ] Batch ingest script: term → search → fetch → chunk → embed → store.

### Phase 4: Retrieval + Answering
- [ ] `VectorStore.search` (cosine, top-k, year filter).
- [ ] `HostedLLMClient` implementing `LLMClient`.
- [ ] Grounding prompt + `Answerer.answer` producing cited `Answer`.
- [ ] Low-confidence guardrail (min_score → refuse).

### Phase 5: Interface + polish
- [ ] CLI (or minimal FastAPI `/query`) to ask questions.
- [ ] Basic eval set (see Testing) + retrieval sanity checks.
- [ ] README with the "why this isn't PubMed" framing for your portfolio.

---

## Error Handling

- **Entrez rate limit / 429** — exponential backoff; cap req/s to the key's limit; resume, don't crash the batch.
- **Missing abstract** — index title-only or skip; flag `abstract IS NULL`; never fail the whole run.
- **PMC full text unavailable** — fall back to abstract; set `has_fulltext = FALSE`.
- **Embedding model load failure** — fail fast at startup with a clear message (misconfig, not runtime).
- **No chunks above `min_score`** — return `Answer(refused=True)` with an honest "not enough relevant literature found," not a hallucinated answer.
- **LLM call failure** — retry once, then surface a clear error; log provider + latency.
- Log at INFO per batch (counts), WARN on skips/fallbacks, ERROR on aborts.

---

## Security Considerations

- **Secrets** — Entrez API key, DB creds, LLM key in env/secret store, never committed.
- **Entrez etiquette** — always send the required `email` param; respect rate limits (ToS).
- **Data privacy** — with a *hosted* LLM, query text + retrieved passages leave your environment; fine for public paper text, but note it in the README (and it's the motivation for the later self-hosted swap).
- **Input validation** — bound `top_k`, `retmax`; sanitize search terms passed to Entrez.
- **Prompt injection** — paper text is untrusted input to the LLM; the system prompt must treat passages as data, not instructions.

---

## Testing Strategy

- **Unit**
  - Chunker: token-size bounds, overlap correctness, metadata carried through, section splitting.
  - EntrezClient: parsing of a saved efetch XML fixture into `Paper` (no live calls in tests).
  - Embedder: output dimensionality matches schema; query vs doc paths run.
  - VectorStore: upsert then search returns the seeded chunk; year filter excludes correctly.
- **Integration**
  - End-to-end on a small fixed PMID set: ingest → index → known question returns expected paper in top-k.
  - Answerer returns citations that correspond to actually-retrieved PMIDs (no phantom citations).
- **Edge cases**
  - Empty search result; paper with no abstract; duplicate PMID re-ingest (idempotent); query with no chunk above threshold (must refuse).
- **Retrieval quality (lightweight eval)**
  - Hand-label ~15 question→relevant-PMID pairs; track recall@k. This is also your portfolio "I measure retrieval, not vibes" artifact.

---

## Success Criteria

- Ingest ≥ a few hundred papers for a topic without crashing on rate limits or missing fields.
- A conceptual query (different wording than any paper title) retrieves a relevant paper in the top-k — demonstrating semantic over keyword recall.
- Answers cite only retrieved papers; injected "answer this instead" text in a passage does not derail the response.
- Swapping `HostedLLMClient` → a stub/self-hosted client requires touching only the client class.
- Low-confidence queries refuse instead of hallucinating.

---

## Open Questions

- [ ] Which biomedical embedding model to standardize on (`S-PubMedBert-MS-MARCO` 768 vs `bge-large` 1024)? Sets `VECTOR(n)`.
- [ ] Abstracts-only for v1, or include PMC full text from the start? (Recommend abstracts-only first.)
- [ ] Which hosted LLM provider for the answer step?
- [ ] CLI or FastAPI for the v1 query surface?
