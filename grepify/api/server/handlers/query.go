package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/grepify/server/sidecar"
)

// QueryReq is the POST body for /api/query
type QueryReq struct {
	Query        string  `json:"query"`
	CollectionID int     `json:"collection_id"`
	Mode         string  `json:"mode"`       // hybrid | debate | cluster | temporal
	TopK         int     `json:"top_k"`
	MinKarma     int     `json:"min_karma"`
	Rerank       *bool   `json:"rerank"`
	MmrLambda    float64 `json:"mmr_lambda"`
}

// QueryResponse is the unified response for all query modes.
type QueryResponse struct {
	Query       string          `json:"query"`
	Mode        string          `json:"mode"`
	Answer      json.RawMessage `json:"answer"`      // varies by mode
	Sources     []Source        `json:"sources"`
	ShareID     string          `json:"share_id"`
	ElapsedMs   float64         `json:"elapsed_ms"`
	RetrievalMs float64         `json:"retrieval_ms"`
	LLMMs       float64         `json:"llm_ms"`
}

type Source struct {
	Title     string `json:"title"`
	Subreddit string `json:"subreddit"`
	URL       string `json:"url"`
	Tier      string `json:"tier"`
	Text      string `json:"text"`
}

// Query handles POST /api/query
func (d *Deps) Query(c *fiber.Ctx) error {
	t0 := time.Now()

	var req QueryReq
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid JSON"})
	}
	if req.Query == "" {
		return c.Status(400).JSON(fiber.Map{"error": "query is required"})
	}
	if req.Mode == "" {
		req.Mode = "hybrid"
	}
	if req.TopK == 0 {
		req.TopK = 5
	}

	// Resolve collection's Qdrant name
	qdrantName := "grepify_finance" // fallback for backward compat
	if req.CollectionID > 0 {
		col, err := d.DB.GetCollection(c.Context(), req.CollectionID)
		if err != nil {
			return c.Status(404).JSON(fiber.Map{"error": "collection not found"})
		}
		qdrantName = col.QdrantName
	}

	rerank := true
	if req.Rerank != nil {
		rerank = *req.Rerank
	}
	mmrLambda := req.MmrLambda
	if mmrLambda == 0 {
		mmrLambda = 0.7
	}

	switch req.Mode {
	case "debate":
		return d.queryDebate(c, req, qdrantName, rerank, mmrLambda, t0)
	case "cluster":
		return d.queryCluster(c, req, qdrantName, rerank, mmrLambda, t0)
	case "temporal":
		return d.queryTemporal(c, req, qdrantName, rerank, mmrLambda, t0)
	default:
		return d.queryHybrid(c, req, qdrantName, rerank, mmrLambda, t0)
	}
}

// ============================================================
// mode=hybrid (default)
// ============================================================

func (d *Deps) queryHybrid(c *fiber.Ctx, req QueryReq, collection string, rerank bool, mmrLambda float64, t0 time.Time) error {
	// 1. Retrieve
	t1 := time.Now()
	searchResp, err := d.Sidecar.Search(sidecar.SearchRequest{
		Query:      req.Query,
		Collection: collection,
		TopK:       req.TopK,
		Mode:       "hybrid",
		Rerank:     rerank,
		MmrLambda:  mmrLambda,
		MinScore:   req.MinKarma,
	})
	retrievalMs := float64(time.Since(t1).Milliseconds())
	if err != nil {
		log.Printf("sidecar search error: %v", err)
		return c.Status(502).JSON(fiber.Map{"error": "retrieval failed"})
	}

	// 2. LLM synthesis
	t2 := time.Now()
	prompt := buildHybridPrompt(req.Query, searchResp.Results)
	answer, err := d.LLM.Complete(systemPrompt, prompt)
	llmMs := float64(time.Since(t2).Milliseconds())
	if err != nil {
		log.Printf("llm error: %v", err)
		answer = "[LLM synthesis failed]"
	}

	// 3. Build sources + share
	sources := resultsToSources(searchResp.Results)
	answerJSON, _ := json.Marshal(map[string]string{"text": answer})

	shareID, _ := d.DB.CreateSharedAnswer(c.Context(), intPtr(req.CollectionID), req.Query, "hybrid", answerJSON)

	return c.JSON(QueryResponse{
		Query:       req.Query,
		Mode:        "hybrid",
		Answer:      answerJSON,
		Sources:     sources,
		ShareID:     shareID,
		ElapsedMs:   float64(time.Since(t0).Milliseconds()),
		RetrievalMs: retrievalMs,
		LLMMs:       llmMs,
	})
}

// ============================================================
// mode=debate
// ============================================================

func (d *Deps) queryDebate(c *fiber.Ctx, req QueryReq, collection string, rerank bool, mmrLambda float64, t0 time.Time) error {
	t1 := time.Now()
	searchResp, err := d.Sidecar.Search(sidecar.SearchRequest{
		Query:      req.Query,
		Collection: collection,
		TopK:       20, // more chunks for debate
		Mode:       "hybrid",
		Rerank:     rerank,
		MmrLambda:  mmrLambda,
		MinScore:   req.MinKarma,
	})
	retrievalMs := float64(time.Since(t1).Milliseconds())
	if err != nil {
		return c.Status(502).JSON(fiber.Map{"error": "retrieval failed"})
	}

	t2 := time.Now()
	prompt := buildDebatePrompt(req.Query, searchResp.Results)
	answer, err := d.LLM.CompleteJSON(debateSystemPrompt, prompt)
	llmMs := float64(time.Since(t2).Milliseconds())
	if err != nil {
		log.Printf("llm debate error: %v", err)
		answer = `{"pro":{"summary":"LLM failed","sources":[]},"con":{"summary":"LLM failed","sources":[]}}`
	}

	sources := resultsToSources(searchResp.Results)
	answerJSON := json.RawMessage(answer)

	shareID, _ := d.DB.CreateSharedAnswer(c.Context(), intPtr(req.CollectionID), req.Query, "debate", answerJSON)

	return c.JSON(QueryResponse{
		Query:       req.Query,
		Mode:        "debate",
		Answer:      answerJSON,
		Sources:     sources,
		ShareID:     shareID,
		ElapsedMs:   float64(time.Since(t0).Milliseconds()),
		RetrievalMs: retrievalMs,
		LLMMs:       llmMs,
	})
}

// ============================================================
// mode=cluster
// ============================================================

func (d *Deps) queryCluster(c *fiber.Ctx, req QueryReq, collection string, rerank bool, mmrLambda float64, t0 time.Time) error {
	t1 := time.Now()
	searchResp, err := d.Sidecar.Search(sidecar.SearchRequest{
		Query:      req.Query,
		Collection: collection,
		TopK:       50, // wide retrieval for clustering
		Mode:       "hybrid",
		Rerank:     false, // no need to rerank before clustering
		MmrLambda:  mmrLambda,
		MinScore:   req.MinKarma,
	})
	retrievalMs := float64(time.Since(t1).Milliseconds())
	if err != nil {
		return c.Status(502).JSON(fiber.Map{"error": "retrieval failed"})
	}

	// Cluster the texts via sidecar
	texts := make([]string, 0, len(searchResp.Results))
	for _, r := range searchResp.Results {
		texts = append(texts, r.Text)
	}

	clusterResp, err := d.Sidecar.Cluster(sidecar.ClusterRequest{Texts: texts})
	if err != nil {
		log.Printf("cluster error: %v", err)
		return c.Status(502).JSON(fiber.Map{"error": "clustering failed"})
	}

	// Label each cluster via LLM
	t2 := time.Now()
	prompt := buildClusterPrompt(req.Query, clusterResp.Clusters)
	answer, err := d.LLM.CompleteJSON(clusterSystemPrompt, prompt)
	llmMs := float64(time.Since(t2).Milliseconds())
	if err != nil {
		log.Printf("llm cluster error: %v", err)
		answer = `{"clusters":[]}`
	}

	sources := resultsToSources(searchResp.Results)
	answerJSON := json.RawMessage(answer)

	shareID, _ := d.DB.CreateSharedAnswer(c.Context(), intPtr(req.CollectionID), req.Query, "cluster", answerJSON)

	return c.JSON(QueryResponse{
		Query:       req.Query,
		Mode:        "cluster",
		Answer:      answerJSON,
		Sources:     sources,
		ShareID:     shareID,
		ElapsedMs:   float64(time.Since(t0).Milliseconds()),
		RetrievalMs: retrievalMs,
		LLMMs:       llmMs,
	})
}

// ============================================================
// mode=temporal
// ============================================================

func (d *Deps) queryTemporal(c *fiber.Ctx, req QueryReq, collection string, rerank bool, mmrLambda float64, t0 time.Time) error {
	t1 := time.Now()
	searchResp, err := d.Sidecar.Search(sidecar.SearchRequest{
		Query:      req.Query,
		Collection: collection,
		TopK:       100, // wide retrieval for temporal analysis
		Mode:       "hybrid",
		Rerank:     false,
		MmrLambda:  mmrLambda,
	})
	retrievalMs := float64(time.Since(t1).Milliseconds())
	if err != nil {
		return c.Status(502).JSON(fiber.Map{"error": "retrieval failed"})
	}

	// Build temporal prompt for LLM
	t2 := time.Now()
	prompt := buildTemporalPrompt(req.Query, searchResp.Results)
	answer, err := d.LLM.CompleteJSON(temporalSystemPrompt, prompt)
	llmMs := float64(time.Since(t2).Milliseconds())
	if err != nil {
		log.Printf("llm temporal error: %v", err)
		answer = `{"timeline":[]}`
	}

	sources := resultsToSources(searchResp.Results)
	answerJSON := json.RawMessage(answer)

	shareID, _ := d.DB.CreateSharedAnswer(c.Context(), intPtr(req.CollectionID), req.Query, "temporal", answerJSON)

	return c.JSON(QueryResponse{
		Query:       req.Query,
		Mode:        "temporal",
		Answer:      answerJSON,
		Sources:     sources,
		ShareID:     shareID,
		ElapsedMs:   float64(time.Since(t0).Milliseconds()),
		RetrievalMs: retrievalMs,
		LLMMs:       llmMs,
	})
}

// ============================================================
// Prompts
// ============================================================

const systemPrompt = `You are a knowledgeable assistant. Answer the user's question based ONLY on the Reddit discussions provided. Be specific, practical, and cite sources by their [number]. If the context doesn't contain enough information, say so clearly.`

const debateSystemPrompt = `You analyze Reddit discussions to find BOTH sides of a debate. You must return a JSON object with two keys: "pro" and "con". Each contains "summary" (2-3 paragraphs) and "sources" (array of [number] references). Be balanced and fair.`

const clusterSystemPrompt = `You analyze grouped opinions from Reddit. For each cluster, provide a "label" (short catchy title), "percentage" (of total), and "summary" (1-2 paragraphs). Return JSON: {"clusters": [{"label": "...", "percentage": N, "summary": "...", "source_refs": [1,2]}]}`

const temporalSystemPrompt = `You analyze how Reddit sentiment on a topic has evolved over time. Return JSON: {"timeline": [{"year": 2021, "sentiment": "positive|negative|mixed", "summary": "...", "key_quotes": ["..."]}], "overall_trend": "..."}`

func buildHybridPrompt(query string, results []sidecar.SearchResult) string {
	ctx := ""
	for i, r := range results {
		ctx += fmt.Sprintf("[%d] r/%s — \"%s\"\n%s\n\n", i+1, r.Subreddit, r.ThreadTitle, r.Text)
	}
	return fmt.Sprintf("--- CONTEXT FROM REDDIT ---\n%s--- END CONTEXT ---\n\nUser question: %s\n\nAnswer in 3-5 concise paragraphs. Cite sources inline like [1], [2].", ctx, query)
}

func buildDebatePrompt(query string, results []sidecar.SearchResult) string {
	ctx := ""
	for i, r := range results {
		ctx += fmt.Sprintf("[%d] r/%s — \"%s\"\n%s\n\n", i+1, r.Subreddit, r.ThreadTitle, r.Text)
	}
	return fmt.Sprintf("Topic: %s\n\n--- REDDIT DISCUSSIONS ---\n%s--- END ---\n\nClassify comments as PRO or CON. Return two sides with summaries and source references.", query, ctx)
}

func buildClusterPrompt(query string, clusters []sidecar.ClusterGroup) string {
	ctx := ""
	total := 0
	for _, cl := range clusters {
		total += cl.Size
	}
	for _, cl := range clusters {
		pct := float64(cl.Size) / float64(total) * 100
		ctx += fmt.Sprintf("Cluster %d (%d items, %.0f%%):\n", cl.ID, cl.Size, pct)
		for j, t := range cl.Texts {
			if j >= 3 {
				ctx += fmt.Sprintf("  ... and %d more\n", len(cl.Texts)-3)
				break
			}
			ctx += fmt.Sprintf("  - %s\n", truncate(t, 200))
		}
		ctx += "\n"
	}
	return fmt.Sprintf("Topic: %s\n\n--- OPINION CLUSTERS ---\n%s--- END ---\n\nLabel each cluster and summarize the opinion.", query, ctx)
}

func buildTemporalPrompt(query string, results []sidecar.SearchResult) string {
	ctx := ""
	for i, r := range results {
		ctx += fmt.Sprintf("[%d] r/%s — \"%s\"\n%s\n\n", i+1, r.Subreddit, r.ThreadTitle, r.Text)
	}
	return fmt.Sprintf("Topic: %s\n\n--- REDDIT DISCUSSIONS (various dates) ---\n%s--- END ---\n\nAnalyze how opinions on this topic evolved over time. Group by year.", query, ctx)
}

// ============================================================
// Helpers
// ============================================================

func resultsToSources(results []sidecar.SearchResult) []Source {
	sources := make([]Source, 0, len(results))
	for _, r := range results {
		sources = append(sources, Source{
			Title:     r.ThreadTitle,
			Subreddit: r.Subreddit,
			URL:       r.URL,
			Tier:      r.Tier,
			Text:      r.Text,
		})
	}
	return sources
}

func intPtr(i int) *int {
	if i == 0 {
		return nil
	}
	return &i
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "…"
}
