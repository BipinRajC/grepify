-- Grepify PostgreSQL schema
-- Applied automatically via docker-entrypoint-initdb.d on first run

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- DOMAINS  (curated subreddit presets)
-- ============================================================
CREATE TABLE domains (
    id          SERIAL PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,          -- "finance", "fitness"
    label       TEXT NOT NULL,                 -- "Finance & Investing"
    description TEXT DEFAULT '',
    icon        TEXT DEFAULT '',               -- emoji or icon name
    subreddits  TEXT[] NOT NULL DEFAULT '{}',  -- {"IndiaInvestments","FIREIndia"}
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- COLLECTIONS  (user-created or domain-activated groups)
-- ============================================================
CREATE TYPE collection_status AS ENUM ('pending', 'indexing', 'ready', 'failed');

CREATE TABLE collections (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT UNIQUE NOT NULL,               -- URL-safe
    description     TEXT DEFAULT '',
    domain          TEXT DEFAULT '',                     -- FK-ish ref to domains.name
    status          collection_status DEFAULT 'pending',
    qdrant_name     TEXT NOT NULL,                       -- "grepify_42"
    thread_count    INT DEFAULT 0,
    chunk_count     INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE collection_subreddits (
    id              SERIAL PRIMARY KEY,
    collection_id   INT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    subreddit       TEXT NOT NULL,
    thread_count    INT DEFAULT 0,
    last_scraped_at TIMESTAMPTZ,
    UNIQUE (collection_id, subreddit)
);

-- ============================================================
-- THREADS + COMMENTS  (raw scraped data)
-- ============================================================
CREATE TABLE threads (
    id              SERIAL PRIMARY KEY,
    reddit_id       TEXT UNIQUE NOT NULL,
    subreddit       TEXT NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT DEFAULT '',
    score           INT DEFAULT 0,
    author          TEXT DEFAULT '[deleted]',
    url             TEXT DEFAULT '',
    created_utc     DOUBLE PRECISION DEFAULT 0,
    num_comments    INT DEFAULT 0,
    scraped_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_threads_subreddit ON threads(subreddit);
CREATE INDEX idx_threads_reddit_id ON threads(reddit_id);

CREATE TABLE comments (
    id              SERIAL PRIMARY KEY,
    reddit_id       TEXT NOT NULL,
    thread_id       INT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    body            TEXT NOT NULL,
    score           INT DEFAULT 0,
    author          TEXT DEFAULT '[deleted]',
    parent_reddit_id TEXT,
    depth           INT DEFAULT 0,
    created_utc     DOUBLE PRECISION DEFAULT 0,
    UNIQUE (reddit_id, thread_id)
);

CREATE INDEX idx_comments_thread ON comments(thread_id);

-- ============================================================
-- CHUNKS  (processed text segments stored alongside Qdrant vectors)
-- ============================================================
CREATE TABLE chunks (
    id              SERIAL PRIMARY KEY,
    chunk_id        TEXT NOT NULL,                       -- "s_abc123", "c_abc_def"
    qdrant_point_id UUID DEFAULT uuid_generate_v4(),
    collection_id   INT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    thread_id       INT REFERENCES threads(id) ON DELETE SET NULL,
    tier            TEXT NOT NULL,                       -- "summary" | "chain" | "insight"
    text            TEXT NOT NULL,
    subreddit       TEXT DEFAULT '',
    thread_title    TEXT DEFAULT '',
    url             TEXT DEFAULT '',
    score           INT DEFAULT 0,                       -- thread or comment score for karma filtering
    sentiment_score DOUBLE PRECISION,                    -- -1.0 to 1.0 via VADER
    year            INT,                                 -- extracted from created_utc for temporal
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_chunks_collection ON chunks(collection_id);
CREATE INDEX idx_chunks_year ON chunks(year);
CREATE INDEX idx_chunks_score ON chunks(score);

-- ============================================================
-- JOBS  (background scraping / indexing tasks)
-- ============================================================
CREATE TYPE job_status AS ENUM ('pending', 'running', 'done', 'failed');
CREATE TYPE job_type   AS ENUM ('index', 'refresh');

CREATE TABLE jobs (
    id              SERIAL PRIMARY KEY,
    collection_id   INT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    type            job_type NOT NULL DEFAULT 'index',
    status          job_status NOT NULL DEFAULT 'pending',
    progress        JSONB DEFAULT '{}',                  -- {"stage":"scraping","subreddit":"X","done":5,"total":50}
    error           TEXT DEFAULT '',
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_jobs_status ON jobs(status);

-- ============================================================
-- SHARED ANSWERS  (cached query results with shareable URLs)
-- ============================================================
CREATE TABLE shared_answers (
    id              SERIAL PRIMARY KEY,
    share_id        UUID UNIQUE DEFAULT uuid_generate_v4(),
    collection_id   INT REFERENCES collections(id) ON DELETE SET NULL,
    query           TEXT NOT NULL,
    mode            TEXT DEFAULT 'hybrid',
    answer          JSONB NOT NULL,                      -- full response payload
    view_count      INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_shared_share_id ON shared_answers(share_id);

-- ============================================================
-- SEED DATA  (curated domain presets)
-- ============================================================
INSERT INTO domains (name, label, description, icon, subreddits) VALUES
    ('finance', 'Finance & Investing', 'Indian personal finance, FIRE, value investing', '💰',
     ARRAY['IndiaInvestments', 'FIREIndia', 'personalfinance', 'fatFIRE', 'ValueInvesting', 'IndianStreetBets']),
    ('fitness', 'Health & Fitness', 'Fitness, bodyweight training, nutrition', '💪',
     ARRAY['fitness', 'xxfitness', 'bodyweightfitness', 'naturalbodybuilding', 'nutrition']),
    ('geopolitics', 'Geopolitics', 'World affairs, defense analysis, geopolitical strategy', '🌍',
     ARRAY['geopolitics', 'worldnews', 'CredibleDefense', 'IndianDefense']),
    ('cs', 'Computer Science & Careers', 'CS careers, experienced devs, ML, learning', '💻',
     ARRAY['cscareerquestions', 'ExperiencedDevs', 'MachineLearning', 'learnprogramming', 'LocalLLaMA']);
