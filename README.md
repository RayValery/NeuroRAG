# NeuroRAG

Semantic search and grounded question-answering over the neuroscience literature.

NeuroRAG ingests papers from PubMed/PMC, indexes them for **meaning-based** retrieval in a pgvector store, and uses an LLM to answer questions with citations drawn *only* from the retrieved passages. It is not a keyword search — it surfaces conceptually related work that shares no vocabulary with your query, and synthesizes across papers into a single cited answer.

This is the first component of a larger platform. The roadmap adds served neuroscience ML models (EEG / spike-train decoders), agent orchestration, and MCP tool servers on top of this retrieval core.

---

## Why not just use PubMed?

PubMed is a keyword/metadata engine: it returns a list of documents and you do the reading and synthesis. NeuroRAG adds a layer PubMed doesn't have — and in fact **uses PubMed (via the Entrez API) as its ingestion source**, not a competitor.

- **Semantic retrieval, not keyword matching.** A query like *"motor cortex spike decoding"* surfaces a paper titled *"neural population dynamics for reaching movements"* — same concept, no shared words.
- **Cross-paper synthesis.** Instead of 40 tabs to read, you get one grounded answer: *"three decoding approaches appear here — Kalman filters, LSTMs, CSP+CNN — the last reports the highest accuracy,"* each claim cited.
- **Question answering.** You can ask *"what preprocessing do most motor-imagery studies use?"* — a question, not a keyword query.

For pure "find me papers on X" lookups, PubMed is faster and better. NeuroRAG wins specifically on conceptual recall and synthesis.

---

## Architecture

```
   PubMed/PMC ──► Ingestion ──► Chunker ──► Embedder ──► pgvector store
   (Entrez API)                                                │
                                                               ▼
   user query ──► embed query ──► vector search (top-k, filters)
                                                               │
                                                               ▼
                                    LLM answerer  ──►  grounded, cited answer
                                (behind swappable LLMClient)
```

| Component | Responsibility |
|-----------|----------------|
| `EntrezClient` | Search + fetch papers from PubMed/PMC, rate-limited |
| `Chunker` | Structure-aware split into ~300–500 token passages with metadata |
| `Embedder` | Encode passages and queries with a biomedical model |
| `VectorStore` | pgvector-backed upsert + cosine similarity search |
| `LLMClient` | Swappable boundary — hosted API now, self-hosted later |
| `Answerer` | Embed query → retrieve → prompt → cited answer |

The `LLMClient` interface is deliberate: swapping a hosted API for a self-hosted vLLM/Ollama endpoint is a one-file change.

---

## Tech stack

- **Python**
- **PostgreSQL + pgvector** — vector store with SQL metadata filtering
- **sentence-transformers** — local embeddings (`pritamdeka/S-PubMedBert-MS-MARCO`, 768-dim) <!-- adjust if you pick bge-large (1024-dim) -->
- **Entrez E-utilities** — PubMed/PMC ingestion
- **A hosted LLM** for the answer step <!-- provider TBD -->
- **Docker Compose** — Postgres + pgvector

---

## Getting started

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- An NCBI account email (required by Entrez); an API key is optional but raises the rate limit from 3 to 10 req/s
- An API key for your chosen LLM provider

### Setup

```bash
git clone <your-repo-url> neurorag
cd neurorag

# Start Postgres + pgvector and apply the schema
docker compose up -d

# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# then edit .env
```

### Environment variables

```
ENTREZ_EMAIL=you@example.com      # required by NCBI
ENTREZ_API_KEY=                   # optional, raises rate limit
DATABASE_URL=postgresql://neurorag:neurorag@localhost:5432/neurorag
EMBEDDING_MODEL=pritamdeka/S-PubMedBert-MS-MARCO
LLM_PROVIDER=                     # your chosen provider
LLM_API_KEY=
```

---

## Usage

### Ingest papers for a topic

```bash
python -m neurorag.ingest --term "motor cortex spike decoding" --retmax 300
```

This searches PubMed, fetches metadata (abstracts for v1), chunks, embeds, and stores everything in pgvector. Re-running is idempotent — papers are deduplicated by PMID.

### Ask a question

```bash
python -m neurorag.ask "What decoding methods are used for motor imagery, and which performs best?"
```

Returns a grounded answer with citations to the specific papers each claim came from. If retrieval confidence is too low, it says so rather than guessing.

<!-- Swap the CLI for a FastAPI /query endpoint if you chose that surface -->

---

## Design principles

- **Grounded or silent.** The LLM answers only from retrieved passages; low retrieval confidence triggers a refusal, not a hallucination.
- **Passages are data, not instructions.** Paper text is treated as untrusted input — the system prompt is hardened against prompt injection embedded in documents.
- **Measured, not vibes.** A small hand-labeled eval set tracks recall@k so retrieval quality is a number, not a feeling.
- **Swappable inference.** Hosted and self-hosted LLMs sit behind one interface.

---

## Project status

**Phase 1 of the platform: retrieval core.** Working toward:

- [ ] Infrastructure — Postgres + pgvector, schema, config
- [ ] Ingestion — Entrez search + fetch, idempotent upsert
- [ ] Indexing — chunking, embedding, vector storage
- [ ] Retrieval + answering — cosine search, grounded cited answers, low-confidence guardrail
- [ ] Interface + eval — query surface, recall@k eval set

### Roadmap (later components)
- Served neuroscience ML model (EEG motor-imagery classifier / spike-train decoder) exposed as an analysis tool
- Agent orchestration tying literature retrieval to data analysis
- MCP servers exposing PubMed search, dataset fetch, and analysis as composable tools
- GraphRAG over a knowledge graph of brain regions, methods, and datasets

See [`specs/neuro-literature-rag.md`](specs/neuro-literature-rag.md) for the full technical specification.

---

## License

<!-- Add a license (MIT is a common choice for portfolio projects). -->
