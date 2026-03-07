package handlers

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/grepify/server/db"
)

// CreateCollectionReq is the POST body for creating a collection.
type CreateCollectionReq struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Domain      string   `json:"domain"`
	Subreddits  []string `json:"subreddits"`
	Limit       int      `json:"limit"` // threads per subreddit (default 50)
}

// CreateCollection handles POST /api/collections
func (d *Deps) CreateCollection(c *fiber.Ctx) error {
	var req CreateCollectionReq
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid JSON: " + err.Error()})
	}
	if req.Name == "" || len(req.Subreddits) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "name and subreddits are required"})
	}
	if req.Limit == 0 {
		req.Limit = 50
	}

	col, err := d.DB.CreateCollection(c.Context(), req.Name, req.Description, req.Domain, req.Subreddits)
	if err != nil {
		log.Printf("create collection error: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to create collection"})
	}

	// Create a background job
	jobID, err := d.DB.CreateJob(c.Context(), col.ID, "index")
	if err != nil {
		log.Printf("create job error: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to create job"})
	}

	// Enqueue the job
	d.Jobs.Enqueue(col.ID, jobID, req.Subreddits, req.Limit)

	return c.Status(201).JSON(fiber.Map{
		"collection": col,
		"job_id":     jobID,
		"message":    "Indexing started — use GET /api/collections/:id/status for live progress",
	})
}

// ListCollections handles GET /api/collections
func (d *Deps) ListCollections(c *fiber.Ctx) error {
	cols, err := d.DB.ListCollections(c.Context())
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	if cols == nil {
		cols = make([]db.Collection, 0)
	}
	return c.JSON(cols)
}

// GetCollection handles GET /api/collections/:id
func (d *Deps) GetCollection(c *fiber.Ctx) error {
	id, err := strconv.Atoi(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	col, err := d.DB.GetCollection(c.Context(), id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "collection not found"})
	}
	return c.JSON(col)
}

// DeleteCollection handles DELETE /api/collections/:id
func (d *Deps) DeleteCollection(c *fiber.Ctx) error {
	id, err := strconv.Atoi(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}

	_, err = d.DB.DeleteCollection(c.Context(), id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "collection not found"})
	}

	// TODO: also delete the Qdrant collection via sidecar/store

	return c.JSON(fiber.Map{"message": "deleted"})
}

// RefreshCollection handles POST /api/collections/:id/refresh
func (d *Deps) RefreshCollection(c *fiber.Ctx) error {
	id, err := strconv.Atoi(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}

	col, err := d.DB.GetCollection(c.Context(), id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "collection not found"})
	}

	var body struct {
		Limit int `json:"limit"`
	}
	_ = c.BodyParser(&body)
	if body.Limit == 0 {
		body.Limit = 50
	}

	jobID, err := d.DB.CreateJob(c.Context(), col.ID, "refresh")
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	d.Jobs.Enqueue(col.ID, jobID, col.Subreddits, body.Limit)

	return c.JSON(fiber.Map{
		"job_id":  jobID,
		"message": "Refresh started",
	})
}

// CollectionStatus handles GET /api/collections/:id/status (SSE)
func (d *Deps) CollectionStatus(c *fiber.Ctx) error {
	id, err := strconv.Atoi(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}

	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")

	c.Context().SetBodyStreamWriter(func(w *bufio.Writer) {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()

		lastProgress := ""
		for range ticker.C {
			job, err := d.DB.GetLatestJob(c.Context(), id)
			if err != nil {
				writeSSE(w, "error", `{"error":"no job found"}`)
				return
			}

			progressStr := string(job.Progress)
			if progressStr != lastProgress || job.Status == "done" || job.Status == "failed" {
				lastProgress = progressStr

				data := map[string]interface{}{
					"job_id": job.ID,
					"status": job.Status,
				}
				if len(job.Progress) > 2 { // not "{}"
					var prog map[string]interface{}
					if json.Unmarshal(job.Progress, &prog) == nil {
						for k, v := range prog {
							data[k] = v
						}
					}
				}
				if job.Error != "" {
					data["error"] = job.Error
				}

				b, _ := json.Marshal(data)
				writeSSE(w, "progress", string(b))
			}

			if job.Status == "done" || job.Status == "failed" {
				writeSSE(w, job.Status, lastProgress)
				return
			}
		}
	})

	return nil
}

func writeSSE(w *bufio.Writer, event, data string) {
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
	_ = w.Flush()
}
