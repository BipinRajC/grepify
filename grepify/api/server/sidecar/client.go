// Package sidecar provides a typed HTTP client for the Python FastAPI ML sidecar.
package sidecar

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

var httpClient = &http.Client{Timeout: 300 * time.Second} // long for embedding batches

// Client talks to the Python FastAPI sidecar.
type Client struct {
	BaseURL string
}

func New(baseURL string) *Client {
	return &Client{BaseURL: baseURL}
}

// ============================================================
// Health
// ============================================================

type HealthResponse struct {
	Status   string `json:"status"`
	BM25Docs int    `json:"bm25_docs"`
	Device   string `json:"device"`
}

func (c *Client) Health() (*HealthResponse, error) {
	var out HealthResponse
	return &out, c.get("/health", &out)
}

// ============================================================
// Scrape
// ============================================================

type ScrapeRequest struct {
	Subreddit  string `json:"subreddit"`
	Sort       string `json:"sort"`
	TimeFilter string `json:"time_filter"`
	Limit      int    `json:"limit"`
}

type ScrapeThread struct {
	ID          string           `json:"id"`
	Subreddit   string           `json:"subreddit"`
	Title       string           `json:"title"`
	Body        string           `json:"body"`
	Score       int              `json:"score"`
	Author      string           `json:"author"`
	URL         string           `json:"url"`
	CreatedUTC  float64          `json:"created_utc"`
	NumComments int              `json:"num_comments"`
	Comments    []ScrapeComment  `json:"comments"`
}

type ScrapeComment struct {
	ID             string          `json:"id"`
	Body           string          `json:"body"`
	Score          int             `json:"score"`
	Author         string          `json:"author"`
	CreatedUTC     float64         `json:"created_utc"`
	ParentID       string          `json:"parent_id"`
	Depth          int             `json:"depth"`
	Replies        []ScrapeComment `json:"replies"`
}

type ScrapeResponse struct {
	Threads   []ScrapeThread `json:"threads"`
	ElapsedMs float64        `json:"elapsed_ms"`
}

func (c *Client) Scrape(req ScrapeRequest) (*ScrapeResponse, error) {
	var out ScrapeResponse
	return &out, c.post("/scrape", req, &out)
}

// ============================================================
// Chunk
// ============================================================

type ChunkRequest struct {
	Threads []ScrapeThread `json:"threads"`
}

type ChunkResult struct {
	ChunkID     string `json:"chunk_id"`
	Text        string `json:"text"`
	Tier        string `json:"tier"`
	ThreadID    string `json:"thread_id"`
	ThreadTitle string `json:"thread_title"`
	Subreddit   string `json:"subreddit"`
	URL         string `json:"url"`
	Score       int    `json:"score"`
	CreatedDate string `json:"created_date"`
}

type ChunkResponse struct {
	Chunks    []ChunkResult `json:"chunks"`
	ElapsedMs float64       `json:"elapsed_ms"`
}

func (c *Client) Chunk(req ChunkRequest) (*ChunkResponse, error) {
	var out ChunkResponse
	return &out, c.post("/chunk", req, &out)
}

// ============================================================
// Embed
// ============================================================

type EmbedRequest struct {
	Texts []string `json:"texts"`
	Kind  string   `json:"kind"` // "query" or "passage"
}

type EmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
	ElapsedMs  float64     `json:"elapsed_ms"`
}

func (c *Client) Embed(req EmbedRequest) (*EmbedResponse, error) {
	var out EmbedResponse
	return &out, c.post("/embed", req, &out)
}

// ============================================================
// Sentiment
// ============================================================

type SentimentRequest struct {
	Texts []string `json:"texts"`
}

type SentimentResponse struct {
	Scores    []float64 `json:"scores"`
	ElapsedMs float64   `json:"elapsed_ms"`
}

func (c *Client) Sentiment(req SentimentRequest) (*SentimentResponse, error) {
	var out SentimentResponse
	return &out, c.post("/sentiment", req, &out)
}

// ============================================================
// Search (full retrieval pipeline)
// ============================================================

type SearchRequest struct {
	Query      string  `json:"query"`
	Collection string  `json:"collection"`
	TopK       int     `json:"top_k"`
	Mode       string  `json:"mode"`
	Rerank     bool    `json:"rerank"`
	MmrLambda  float64 `json:"mmr_lambda"`
	MinScore   int     `json:"min_score"`  // karma filter
}

type SearchResult struct {
	Text         string   `json:"text"`
	Tier         string   `json:"tier"`
	ThreadID     string   `json:"thread_id"`
	ThreadTitle  string   `json:"thread_title"`
	Subreddit    string   `json:"subreddit"`
	URL          string   `json:"url"`
	Score        float64  `json:"score"`
	RerankScore  *float64 `json:"rerank_score"`
}

type SearchResponse struct {
	Query     string         `json:"query"`
	Results   []SearchResult `json:"results"`
	ElapsedMs float64        `json:"elapsed_ms"`
}

func (c *Client) Search(req SearchRequest) (*SearchResponse, error) {
	var out SearchResponse
	return &out, c.post("/search", req, &out)
}

// ============================================================
// Cluster
// ============================================================

type ClusterRequest struct {
	Texts []string `json:"texts"`
}

type ClusterGroup struct {
	ID    int      `json:"id"`
	Texts []string `json:"texts"`
	Size  int      `json:"size"`
}

type ClusterResponse struct {
	Clusters  []ClusterGroup `json:"clusters"`
	ElapsedMs float64        `json:"elapsed_ms"`
}

func (c *Client) Cluster(req ClusterRequest) (*ClusterResponse, error) {
	var out ClusterResponse
	return &out, c.post("/cluster", req, &out)
}

// ============================================================
// Upload (batch upsert to Qdrant)
// ============================================================

type UploadPoint struct {
	ID      string                 `json:"id"`
	Vector  []float64              `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
}

type UploadRequest struct {
	Collection string        `json:"collection"`
	Points     []UploadPoint `json:"points"`
}

type UploadResponse struct {
	Uploaded  int     `json:"uploaded"`
	ElapsedMs float64 `json:"elapsed_ms"`
}

func (c *Client) Upload(req UploadRequest) error {
	var out UploadResponse
	return c.post("/upload", req, &out)
}

// ============================================================
// Ensure Collection
// ============================================================

type EnsureCollectionReq struct {
	Name string `json:"name"`
	Dim  int    `json:"dim"`
}

type EnsureCollectionResp struct {
	Name    string `json:"name"`
	Created bool   `json:"created"`
}

func (c *Client) EnsureCollection(req EnsureCollectionReq) error {
	var out EnsureCollectionResp
	return c.post("/collections/ensure", req, &out)
}

// ============================================================
// HTTP helpers
// ============================================================

func (c *Client) get(path string, target any) error {
	resp, err := httpClient.Get(c.BaseURL + path)
	if err != nil {
		return fmt.Errorf("sidecar GET %s: %w", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("sidecar GET %s: HTTP %d: %s", path, resp.StatusCode, string(b))
	}
	return json.NewDecoder(resp.Body).Decode(target)
}

func (c *Client) post(path string, payload, target any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	resp, err := httpClient.Post(c.BaseURL+path, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("sidecar POST %s: %w", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("sidecar POST %s: HTTP %d: %s", path, resp.StatusCode, string(b))
	}
	return json.NewDecoder(resp.Body).Decode(target)
}
