CREATE EXTENSION IF NOT EXISTS vector;

-- Papers: one row per PMID
CREATE TABLE IF NOT EXISTS papers (
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
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     BIGSERIAL PRIMARY KEY,
    pmid     TEXT REFERENCES papers(pmid) ON DELETE CASCADE,
    section     TEXT,            -- 'abstract' | 'methods' | 'results' | ...
    chunk_index INT,             -- order within the paper
    content     TEXT NOT NULL,
    embedding   VECTOR(768)      -- vector of numbers (coordinates of chunk in [768]-dimensional space); dimension must match the embedding model
);

-- ANN index for cosine similarity
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);    -- hnsw = Hierarchical Navigable Small World algorithm of indexing;
                                                                    -- vector_cosine_ops - distance metric measures angles instead of length, should be compared with operator <=>
CREATE INDEX ON papers (year);