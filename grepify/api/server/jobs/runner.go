// Package jobs provides a background worker that drives the
// scrape → chunk → embed → upload → sentiment pipeline.
package jobs

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"

	"github.com/grepify/server/db"
	"github.com/grepify/server/sidecar"
)

// Runner manages a fixed-size goroutine pool for indexing jobs.
type Runner struct {
	DB      *db.Pool
	Sidecar *sidecar.Client
	ch      chan work
	wg      sync.WaitGroup
}

type work struct {
	CollectionID int
	JobID        int
	Subreddits   []string
	Limit        int
}

// New creates a runner with `workers` goroutines.
func New(dbPool *db.Pool, sc *sidecar.Client, workers int) *Runner {
	if workers <= 0 {
		workers = 2
	}
	r := &Runner{
		DB:      dbPool,
		Sidecar: sc,
		ch:      make(chan work, 100),
	}
	for i := 0; i < workers; i++ {
		r.wg.Add(1)
		go r.loop(i)
	}
	return r
}

// Enqueue submits a new indexing job.
func (r *Runner) Enqueue(collectionID, jobID int, subreddits []string, limit int) {
	r.ch <- work{
		CollectionID: collectionID,
		JobID:        jobID,
		Subreddits:   subreddits,
		Limit:        limit,
	}
}

// Shutdown drains the channel and waits for workers.
func (r *Runner) Shutdown() {
	close(r.ch)
	r.wg.Wait()
}

func (r *Runner) loop(workerID int) {
	defer r.wg.Done()
	for w := range r.ch {
		log.Printf("[worker %d] starting job %d (collection %d)", workerID, w.JobID, w.CollectionID)
		if err := r.run(w); err != nil {
			log.Printf("[worker %d] job %d FAILED: %v", workerID, w.JobID, err)
			_ = r.DB.UpdateJobError(context.Background(), w.JobID, err.Error())
			_ = r.DB.UpdateCollectionStatus(context.Background(), w.CollectionID, "error")
		} else {
			log.Printf("[worker %d] job %d DONE", workerID, w.JobID)
		}
	}
}

// Progress tracks indexing state and is stored as JSONB in the jobs table.
type Progress struct {
	Stage       string `json:"stage"`
	Subreddit   string `json:"subreddit,omitempty"`
	SubIdx      int    `json:"sub_idx"`
	SubTotal    int    `json:"sub_total"`
	Threads     int    `json:"threads"`
	Chunks      int    `json:"chunks"`
	Embedded    int    `json:"embedded"`
	Uploaded    int    `json:"uploaded"`
	Message     string `json:"message,omitempty"`
}

func (r *Runner) setProgress(jobID int, p Progress) {
	data, _ := json.Marshal(p)
	_ = r.DB.UpdateJobProgress(context.Background(), jobID, data)
}

func (r *Runner) run(w work) error {
	ctx := context.Background()
	_ = r.DB.UpdateJobStatus(ctx, w.JobID, "running")
	_ = r.DB.UpdateCollectionStatus(ctx, w.CollectionID, "indexing")

	// Resolve Qdrant collection name
	col, err := r.DB.GetCollection(ctx, w.CollectionID)
	if err != nil {
		return fmt.Errorf("get collection: %w", err)
	}

	// Ensure Qdrant collection exists
	if err := r.Sidecar.EnsureCollection(sidecar.EnsureCollectionReq{Name: col.QdrantName, Dim: 768}); err != nil {
		return fmt.Errorf("ensure qdrant collection: %w", err)
	}

	totalThreads := 0
	totalChunks := 0

	for i, sub := range w.Subreddits {
		prog := Progress{
			Stage:    "scraping",
			Subreddit: sub,
			SubIdx:   i + 1,
			SubTotal: len(w.Subreddits),
			Message:  fmt.Sprintf("Scraping r/%s...", sub),
		}
		r.setProgress(w.JobID, prog)

		// 1. Scrape
		scrapeResp, err := r.Sidecar.Scrape(sidecar.ScrapeRequest{
			Subreddit:  sub,
			Sort:       "top",
			TimeFilter: "all",
			Limit:      w.Limit,
		})
		if err != nil {
			return fmt.Errorf("scrape r/%s: %w", sub, err)
		}
		prog.Threads = len(scrapeResp.Threads)
		prog.Stage = "saving"
		prog.Message = fmt.Sprintf("Saving %d threads from r/%s", len(scrapeResp.Threads), sub)
		r.setProgress(w.JobID, prog)
		totalThreads += len(scrapeResp.Threads)

		// 2. Save threads + comments to Postgres
		for _, t := range scrapeResp.Threads {
			threadID, err := r.DB.UpsertThread(ctx, db.Thread{
				RedditID:    t.ID,
				Subreddit:   t.Subreddit,
				Title:       t.Title,
				Body:        t.Body,
				Score:       t.Score,
				Author:      t.Author,
				URL:         t.URL,
				CreatedUTC:  t.CreatedUTC,
				NumComments: t.NumComments,
			})
			if err != nil {
				log.Printf("upsert thread %s: %v", t.ID, err)
				continue
			}

			// Flatten comments
			comments := flattenComments(t.Comments, threadID)
			if len(comments) > 0 {
				if err := r.DB.BulkInsertComments(ctx, comments); err != nil {
					log.Printf("insert comments for thread %s: %v", t.ID, err)
				}
			}
		}

		// Update subreddit scrape stats
		_ = r.DB.UpdateSubredditScraped(ctx, w.CollectionID, sub, len(scrapeResp.Threads))

		// 3. Chunk
		prog.Stage = "chunking"
		prog.Message = fmt.Sprintf("Chunking %d threads from r/%s", len(scrapeResp.Threads), sub)
		r.setProgress(w.JobID, prog)

		chunkResp, err := r.Sidecar.Chunk(sidecar.ChunkRequest{Threads: scrapeResp.Threads})
		if err != nil {
			return fmt.Errorf("chunk r/%s: %w", sub, err)
		}
		prog.Chunks = len(chunkResp.Chunks)
		totalChunks += len(chunkResp.Chunks)

		if len(chunkResp.Chunks) == 0 {
			continue
		}

		// 4. Embed (batch)
		prog.Stage = "embedding"
		prog.Message = fmt.Sprintf("Embedding %d chunks from r/%s", len(chunkResp.Chunks), sub)
		r.setProgress(w.JobID, prog)

		texts := make([]string, len(chunkResp.Chunks))
		for j, ch := range chunkResp.Chunks {
			texts[j] = ch.Text
		}

		embedResp, err := r.Sidecar.Embed(sidecar.EmbedRequest{Texts: texts, Kind: "passage"})
		if err != nil {
			return fmt.Errorf("embed r/%s: %w", sub, err)
		}
		prog.Embedded = len(embedResp.Embeddings)

		// 5. Sentiment (batch)
		prog.Stage = "sentiment"
		prog.Message = fmt.Sprintf("Scoring sentiment for %d chunks from r/%s", len(texts), sub)
		r.setProgress(w.JobID, prog)

		sentResp, err := r.Sidecar.Sentiment(sidecar.SentimentRequest{Texts: texts})
		if err != nil {
			log.Printf("sentiment error for r/%s (non-fatal): %v", sub, err)
			sentResp = &sidecar.SentimentResponse{Scores: make([]float64, len(texts))}
		}

		// 6. Save chunks to Postgres & prepare Qdrant upload
		prog.Stage = "uploading"
		prog.Message = fmt.Sprintf("Uploading %d vectors for r/%s", len(embedResp.Embeddings), sub)
		r.setProgress(w.JobID, prog)

		dbChunks := make([]db.Chunk, len(chunkResp.Chunks))
		for j, ch := range chunkResp.Chunks {
			var sentScore *float64
			if j < len(sentResp.Scores) {
				s := sentResp.Scores[j]
				sentScore = &s
			}
			var year *int
			if ch.CreatedDate != "" && len(ch.CreatedDate) >= 4 {
				y := 0
				fmt.Sscanf(ch.CreatedDate[:4], "%d", &y)
				if y > 2000 {
					year = &y
				}
			}
			dbChunks[j] = db.Chunk{
				ChunkID:        ch.ChunkID,
				CollectionID:   w.CollectionID,
				Tier:           ch.Tier,
				Text:           ch.Text,
				Subreddit:      ch.Subreddit,
				ThreadTitle:    ch.ThreadTitle,
				URL:            ch.URL,
				Score:          ch.Score,
				SentimentScore: sentScore,
				Year:           year,
			}
		}

		pointIDs, err := r.DB.BulkInsertChunks(ctx, dbChunks)
		if err != nil {
			return fmt.Errorf("insert chunks r/%s: %w", sub, err)
		}

		// 7. Upload to Qdrant via sidecar
		uploadPoints := make([]sidecar.UploadPoint, len(pointIDs))
		for j := range pointIDs {
			payload := map[string]interface{}{
				"text":         chunkResp.Chunks[j].Text,
				"tier":         chunkResp.Chunks[j].Tier,
				"thread_id":    chunkResp.Chunks[j].ThreadID,
				"thread_title": chunkResp.Chunks[j].ThreadTitle,
				"subreddit":    chunkResp.Chunks[j].Subreddit,
				"url":          chunkResp.Chunks[j].URL,
				"score":        chunkResp.Chunks[j].Score,
				"chunk_id":     chunkResp.Chunks[j].ChunkID,
			}
			if j < len(sentResp.Scores) {
				payload["sentiment_score"] = sentResp.Scores[j]
			}
			uploadPoints[j] = sidecar.UploadPoint{
				ID:      pointIDs[j],
				Vector:  embedResp.Embeddings[j],
				Payload: payload,
			}
		}

		if err := r.Sidecar.Upload(sidecar.UploadRequest{
			Collection: col.QdrantName,
			Points:     uploadPoints,
		}); err != nil {
			return fmt.Errorf("upload r/%s: %w", sub, err)
		}

		prog.Uploaded = len(uploadPoints)
		prog.Message = fmt.Sprintf("Done with r/%s: %d threads, %d chunks", sub, len(scrapeResp.Threads), len(chunkResp.Chunks))
		r.setProgress(w.JobID, prog)
	}

	// Finalise
	_ = r.DB.UpdateCollectionCounts(ctx, w.CollectionID, totalThreads, totalChunks)
	_ = r.DB.UpdateCollectionStatus(ctx, w.CollectionID, "ready")
	_ = r.DB.UpdateJobStatus(ctx, w.JobID, "done")

	finalProg := Progress{
		Stage:    "done",
		SubIdx:   len(w.Subreddits),
		SubTotal: len(w.Subreddits),
		Threads:  totalThreads,
		Chunks:   totalChunks,
		Uploaded: totalChunks,
		Message:  fmt.Sprintf("Complete: %d threads, %d chunks indexed", totalThreads, totalChunks),
	}
	r.setProgress(w.JobID, finalProg)

	return nil
}

// flattenComments recursively flattens nested comments.
func flattenComments(comments []sidecar.ScrapeComment, threadID int) []db.Comment {
	var flat []db.Comment
	for _, c := range comments {
		flat = append(flat, db.Comment{
			RedditID:       c.ID,
			ThreadID:       threadID,
			Body:           c.Body,
			Score:          c.Score,
			Author:         c.Author,
			ParentRedditID: c.ParentID,
			Depth:          c.Depth,
			CreatedUTC:     c.CreatedUTC,
		})
		if len(c.Replies) > 0 {
			flat = append(flat, flattenComments(c.Replies, threadID)...)
		}
	}
	return flat
}
