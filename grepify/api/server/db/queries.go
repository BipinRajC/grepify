package db

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ============================================================
// Model types
// ============================================================

type Collection struct {
	ID          int       `json:"id"`
	Name        string    `json:"name"`
	Slug        string    `json:"slug"`
	Description string    `json:"description"`
	Domain      string    `json:"domain"`
	Status      string    `json:"status"`
	QdrantName  string    `json:"qdrant_name"`
	ThreadCount int       `json:"thread_count"`
	ChunkCount  int       `json:"chunk_count"`
	Subreddits  []string  `json:"subreddits"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type CollectionSubreddit struct {
	ID            int        `json:"id"`
	CollectionID  int        `json:"collection_id"`
	Subreddit     string     `json:"subreddit"`
	ThreadCount   int        `json:"thread_count"`
	LastScrapedAt *time.Time `json:"last_scraped_at"`
}

type Thread struct {
	ID          int     `json:"id"`
	RedditID    string  `json:"reddit_id"`
	Subreddit   string  `json:"subreddit"`
	Title       string  `json:"title"`
	Body        string  `json:"body"`
	Score       int     `json:"score"`
	Author      string  `json:"author"`
	URL         string  `json:"url"`
	CreatedUTC  float64 `json:"created_utc"`
	NumComments int     `json:"num_comments"`
}

type Comment struct {
	ID             int     `json:"id"`
	RedditID       string  `json:"reddit_id"`
	ThreadID       int     `json:"thread_id"`
	Body           string  `json:"body"`
	Score          int     `json:"score"`
	Author         string  `json:"author"`
	ParentRedditID string  `json:"parent_reddit_id"`
	Depth          int     `json:"depth"`
	CreatedUTC     float64 `json:"created_utc"`
}

type Chunk struct {
	ID             int      `json:"id"`
	ChunkID        string   `json:"chunk_id"`
	QdrantPointID  string   `json:"qdrant_point_id"`
	CollectionID   int      `json:"collection_id"`
	ThreadID       *int     `json:"thread_id"`
	Tier           string   `json:"tier"`
	Text           string   `json:"text"`
	Subreddit      string   `json:"subreddit"`
	ThreadTitle    string   `json:"thread_title"`
	URL            string   `json:"url"`
	Score          int      `json:"score"`
	SentimentScore *float64 `json:"sentiment_score"`
	Year           *int     `json:"year"`
}

type Job struct {
	ID           int              `json:"id"`
	CollectionID int              `json:"collection_id"`
	Type         string           `json:"type"`
	Status       string           `json:"status"`
	Progress     json.RawMessage  `json:"progress"`
	Error        string           `json:"error"`
	StartedAt    *time.Time       `json:"started_at"`
	CompletedAt  *time.Time       `json:"completed_at"`
	CreatedAt    time.Time        `json:"created_at"`
}

type SharedAnswer struct {
	ID           int             `json:"id"`
	ShareID      string          `json:"share_id"`
	CollectionID *int            `json:"collection_id"`
	Query        string          `json:"query"`
	Mode         string          `json:"mode"`
	Answer       json.RawMessage `json:"answer"`
	ViewCount    int             `json:"view_count"`
	CreatedAt    time.Time       `json:"created_at"`
}

type Domain struct {
	ID          int      `json:"id"`
	Name        string   `json:"name"`
	Label       string   `json:"label"`
	Description string   `json:"description"`
	Icon        string   `json:"icon"`
	Subreddits  []string `json:"subreddits"`
}

// ============================================================
// Collections
// ============================================================

func (p *Pool) CreateCollection(ctx context.Context, name, description, domain string, subreddits []string) (*Collection, error) {
	slug := slugify(name)
	qdrantName := fmt.Sprintf("grepify_%s", slug)

	var col Collection
	err := p.QueryRow(ctx,
		`INSERT INTO collections (name, slug, description, domain, qdrant_name)
		 VALUES ($1, $2, $3, $4, $5)
		 RETURNING id, name, slug, description, domain, status, qdrant_name, thread_count, chunk_count, created_at, updated_at`,
		name, slug, description, domain, qdrantName,
	).Scan(&col.ID, &col.Name, &col.Slug, &col.Description, &col.Domain,
		&col.Status, &col.QdrantName, &col.ThreadCount, &col.ChunkCount,
		&col.CreatedAt, &col.UpdatedAt)
	if err != nil {
		return nil, fmt.Errorf("insert collection: %w", err)
	}
	col.Subreddits = subreddits

	// Insert subreddits
	for _, sub := range subreddits {
		_, err := p.Exec(ctx,
			`INSERT INTO collection_subreddits (collection_id, subreddit) VALUES ($1, $2)
			 ON CONFLICT DO NOTHING`,
			col.ID, sub,
		)
		if err != nil {
			return nil, fmt.Errorf("insert subreddit %s: %w", sub, err)
		}
	}

	return &col, nil
}

func (p *Pool) GetCollection(ctx context.Context, id int) (*Collection, error) {
	var col Collection
	err := p.QueryRow(ctx,
		`SELECT id, name, slug, description, domain, status, qdrant_name, thread_count, chunk_count, created_at, updated_at
		 FROM collections WHERE id = $1`, id,
	).Scan(&col.ID, &col.Name, &col.Slug, &col.Description, &col.Domain,
		&col.Status, &col.QdrantName, &col.ThreadCount, &col.ChunkCount,
		&col.CreatedAt, &col.UpdatedAt)
	if err != nil {
		return nil, err
	}

	// Fetch subreddits
	rows, err := p.Query(ctx,
		`SELECT subreddit FROM collection_subreddits WHERE collection_id = $1`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			return nil, err
		}
		col.Subreddits = append(col.Subreddits, s)
	}
	return &col, nil
}

func (p *Pool) ListCollections(ctx context.Context) ([]Collection, error) {
	rows, err := p.Query(ctx,
		`SELECT c.id, c.name, c.slug, c.description, c.domain, c.status, c.qdrant_name,
		        c.thread_count, c.chunk_count, c.created_at, c.updated_at
		 FROM collections c ORDER BY c.created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var cols []Collection
	for rows.Next() {
		var col Collection
		if err := rows.Scan(&col.ID, &col.Name, &col.Slug, &col.Description, &col.Domain,
			&col.Status, &col.QdrantName, &col.ThreadCount, &col.ChunkCount,
			&col.CreatedAt, &col.UpdatedAt); err != nil {
			return nil, err
		}
		cols = append(cols, col)
	}

	// Attach subreddits
	for i := range cols {
		rows2, err := p.Query(ctx,
			`SELECT subreddit FROM collection_subreddits WHERE collection_id = $1`, cols[i].ID)
		if err != nil {
			return nil, err
		}
		for rows2.Next() {
			var s string
			if err := rows2.Scan(&s); err != nil {
				rows2.Close()
				return nil, err
			}
			cols[i].Subreddits = append(cols[i].Subreddits, s)
		}
		rows2.Close()
	}

	return cols, nil
}

func (p *Pool) DeleteCollection(ctx context.Context, id int) (string, error) {
	// Return qdrant_name for cleanup
	var qdrantName string
	err := p.QueryRow(ctx,
		`DELETE FROM collections WHERE id = $1 RETURNING qdrant_name`, id,
	).Scan(&qdrantName)
	return qdrantName, err
}

func (p *Pool) UpdateCollectionStatus(ctx context.Context, id int, status string) error {
	_, err := p.Exec(ctx,
		`UPDATE collections SET status = $2, updated_at = now() WHERE id = $1`,
		id, status)
	return err
}

func (p *Pool) UpdateCollectionCounts(ctx context.Context, id, threads, chunks int) error {
	_, err := p.Exec(ctx,
		`UPDATE collections SET thread_count = $2, chunk_count = $3, updated_at = now() WHERE id = $1`,
		id, threads, chunks)
	return err
}

// ============================================================
// Threads
// ============================================================

func (p *Pool) UpsertThread(ctx context.Context, t Thread) (int, error) {
	var id int
	err := p.QueryRow(ctx,
		`INSERT INTO threads (reddit_id, subreddit, title, body, score, author, url, created_utc, num_comments)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		 ON CONFLICT (reddit_id) DO UPDATE SET score = EXCLUDED.score, num_comments = EXCLUDED.num_comments
		 RETURNING id`,
		t.RedditID, t.Subreddit, t.Title, t.Body, t.Score, t.Author, t.URL, t.CreatedUTC, t.NumComments,
	).Scan(&id)
	return id, err
}

func (p *Pool) GetExistingRedditIDs(ctx context.Context, subreddit string) (map[string]bool, error) {
	rows, err := p.Query(ctx,
		`SELECT reddit_id FROM threads WHERE subreddit = $1`, subreddit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	ids := make(map[string]bool)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids[id] = true
	}
	return ids, nil
}

// ============================================================
// Comments
// ============================================================

func (p *Pool) BulkInsertComments(ctx context.Context, comments []Comment) error {
	if len(comments) == 0 {
		return nil
	}
	// Batch insert via multi-row VALUES
	const batchSize = 200
	for i := 0; i < len(comments); i += batchSize {
		end := i + batchSize
		if end > len(comments) {
			end = len(comments)
		}
		batch := comments[i:end]

		var b strings.Builder
		b.WriteString(`INSERT INTO comments (reddit_id, thread_id, body, score, author, parent_reddit_id, depth, created_utc) VALUES `)
		args := make([]interface{}, 0, len(batch)*8)
		for j, c := range batch {
			if j > 0 {
				b.WriteString(",")
			}
			n := j * 8
			b.WriteString(fmt.Sprintf("($%d,$%d,$%d,$%d,$%d,$%d,$%d,$%d)",
				n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8))
			args = append(args, c.RedditID, c.ThreadID, c.Body, c.Score, c.Author, c.ParentRedditID, c.Depth, c.CreatedUTC)
		}
		b.WriteString(` ON CONFLICT (reddit_id, thread_id) DO NOTHING`)

		if _, err := p.Exec(ctx, b.String(), args...); err != nil {
			return fmt.Errorf("bulk insert comments: %w", err)
		}
	}
	return nil
}

// ============================================================
// Chunks
// ============================================================

func (p *Pool) BulkInsertChunks(ctx context.Context, chunks []Chunk) ([]string, error) {
	pointIDs := make([]string, len(chunks))
	for i, c := range chunks {
		uid := uuid.New().String()
		pointIDs[i] = uid
		_, err := p.Exec(ctx,
			`INSERT INTO chunks (chunk_id, qdrant_point_id, collection_id, thread_id, tier, text, subreddit, thread_title, url, score, sentiment_score, year)
			 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`,
			c.ChunkID, uid, c.CollectionID, c.ThreadID, c.Tier, c.Text,
			c.Subreddit, c.ThreadTitle, c.URL, c.Score, c.SentimentScore, c.Year,
		)
		if err != nil {
			return nil, fmt.Errorf("insert chunk %s: %w", c.ChunkID, err)
		}
	}
	return pointIDs, nil
}

func (p *Pool) GetChunksByCollection(ctx context.Context, collectionID int) ([]Chunk, error) {
	rows, err := p.Query(ctx,
		`SELECT id, chunk_id, qdrant_point_id, collection_id, thread_id, tier, text,
		        subreddit, thread_title, url, score, sentiment_score, year
		 FROM chunks WHERE collection_id = $1`, collectionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var chunks []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.ChunkID, &c.QdrantPointID, &c.CollectionID,
			&c.ThreadID, &c.Tier, &c.Text, &c.Subreddit, &c.ThreadTitle,
			&c.URL, &c.Score, &c.SentimentScore, &c.Year); err != nil {
			return nil, err
		}
		chunks = append(chunks, c)
	}
	return chunks, nil
}

func (p *Pool) UpdateSubredditScraped(ctx context.Context, collectionID int, subreddit string, threadCount int) error {
	_, err := p.Exec(ctx,
		`UPDATE collection_subreddits SET thread_count = $3, last_scraped_at = now()
		 WHERE collection_id = $1 AND subreddit = $2`,
		collectionID, subreddit, threadCount)
	return err
}

// ============================================================
// Jobs
// ============================================================

func (p *Pool) CreateJob(ctx context.Context, collectionID int, jobType string) (int, error) {
	var id int
	err := p.QueryRow(ctx,
		`INSERT INTO jobs (collection_id, type) VALUES ($1, $2) RETURNING id`,
		collectionID, jobType,
	).Scan(&id)
	return id, err
}

func (p *Pool) GetJob(ctx context.Context, id int) (*Job, error) {
	var j Job
	err := p.QueryRow(ctx,
		`SELECT id, collection_id, type, status, progress, error, started_at, completed_at, created_at
		 FROM jobs WHERE id = $1`, id,
	).Scan(&j.ID, &j.CollectionID, &j.Type, &j.Status, &j.Progress, &j.Error,
		&j.StartedAt, &j.CompletedAt, &j.CreatedAt)
	return &j, err
}

func (p *Pool) GetLatestJob(ctx context.Context, collectionID int) (*Job, error) {
	var j Job
	err := p.QueryRow(ctx,
		`SELECT id, collection_id, type, status, progress, error, started_at, completed_at, created_at
		 FROM jobs WHERE collection_id = $1 ORDER BY created_at DESC LIMIT 1`, collectionID,
	).Scan(&j.ID, &j.CollectionID, &j.Type, &j.Status, &j.Progress, &j.Error,
		&j.StartedAt, &j.CompletedAt, &j.CreatedAt)
	return &j, err
}

func (p *Pool) UpdateJobStatus(ctx context.Context, id int, status string) error {
	now := time.Now()
	switch status {
	case "running":
		_, err := p.Exec(ctx, `UPDATE jobs SET status = $2, started_at = $3 WHERE id = $1`, id, status, now)
		return err
	case "done", "failed":
		_, err := p.Exec(ctx, `UPDATE jobs SET status = $2, completed_at = $3 WHERE id = $1`, id, status, now)
		return err
	default:
		_, err := p.Exec(ctx, `UPDATE jobs SET status = $2 WHERE id = $1`, id, status)
		return err
	}
}

func (p *Pool) UpdateJobProgress(ctx context.Context, id int, progress json.RawMessage) error {
	_, err := p.Exec(ctx, `UPDATE jobs SET progress = $2 WHERE id = $1`, id, progress)
	return err
}

func (p *Pool) UpdateJobError(ctx context.Context, id int, errMsg string) error {
	_, err := p.Exec(ctx,
		`UPDATE jobs SET status = 'failed', error = $2, completed_at = now() WHERE id = $1`,
		id, errMsg)
	return err
}

// ============================================================
// Shared Answers
// ============================================================

func (p *Pool) CreateSharedAnswer(ctx context.Context, collectionID *int, query, mode string, answer json.RawMessage) (string, error) {
	shareID := uuid.New().String()
	_, err := p.Exec(ctx,
		`INSERT INTO shared_answers (share_id, collection_id, query, mode, answer) VALUES ($1, $2, $3, $4, $5)`,
		shareID, collectionID, query, mode, answer)
	return shareID, err
}

func (p *Pool) GetSharedAnswer(ctx context.Context, shareID string) (*SharedAnswer, error) {
	var a SharedAnswer
	err := p.QueryRow(ctx,
		`UPDATE shared_answers SET view_count = view_count + 1
		 WHERE share_id = $1
		 RETURNING id, share_id, collection_id, query, mode, answer, view_count, created_at`,
		shareID,
	).Scan(&a.ID, &a.ShareID, &a.CollectionID, &a.Query, &a.Mode, &a.Answer, &a.ViewCount, &a.CreatedAt)
	return &a, err
}

// ============================================================
// Domains
// ============================================================

func (p *Pool) ListDomains(ctx context.Context) ([]Domain, error) {
	rows, err := p.Query(ctx,
		`SELECT id, name, label, description, icon, subreddits FROM domains ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var domains []Domain
	for rows.Next() {
		var d Domain
		if err := rows.Scan(&d.ID, &d.Name, &d.Label, &d.Description, &d.Icon, &d.Subreddits); err != nil {
			return nil, err
		}
		domains = append(domains, d)
	}
	return domains, nil
}

func (p *Pool) GetDomain(ctx context.Context, name string) (*Domain, error) {
	var d Domain
	err := p.QueryRow(ctx,
		`SELECT id, name, label, description, icon, subreddits FROM domains WHERE name = $1`, name,
	).Scan(&d.ID, &d.Name, &d.Label, &d.Description, &d.Icon, &d.Subreddits)
	return &d, err
}

// ============================================================
// Helpers
// ============================================================

func slugify(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			return r
		}
		if r == ' ' || r == '_' {
			return '-'
		}
		return -1
	}, s)
	// collapse multiple dashes
	for strings.Contains(s, "--") {
		s = strings.ReplaceAll(s, "--", "-")
	}
	return strings.Trim(s, "-")
}
