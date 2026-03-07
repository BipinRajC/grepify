// Grepify API server — Go Fiber v2
//
// Multi-domain RAG search engine over Reddit.
// Packages: db, handlers, jobs, llm, sidecar.
//
// Run:
//   go run .   (reads .env from project root)

package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/joho/godotenv"

	"github.com/grepify/server/db"
	"github.com/grepify/server/handlers"
	"github.com/grepify/server/jobs"
	"github.com/grepify/server/llm"
	"github.com/grepify/server/sidecar"
)

func main() {
	// Load .env from project root (two levels up from api/server/)
	_ = godotenv.Load("../../.env")

	ctx := context.Background()

	// ---- Postgres --------------------------------------------------------
	dsn := getenv("DATABASE_URL", "postgres://grepify:grepify_dev@localhost:5432/grepify?sslmode=disable")
	pool, err := db.Connect(ctx, dsn)
	if err != nil {
		log.Fatalf("postgres: %v", err)
	}
	defer pool.Close()
	if err := pool.Migrate(ctx); err != nil {
		log.Fatalf("migration: %v", err)
	}

	// ---- Sidecar ---------------------------------------------------------
	sc := sidecar.New(getenv("SIDECAR_URL", "http://localhost:8001"))

	// ---- LLM -------------------------------------------------------------
	llmClient := llm.NewGroq(
		getenv("GROQ_API_KEY", ""),
		getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
	)

	// ---- Jobs runner (2 workers) -----------------------------------------
	runner := jobs.New(pool, sc, 2)
	defer runner.Shutdown()

	// ---- Deps shared by all handlers -------------------------------------
	deps := &handlers.Deps{
		DB:      pool,
		Sidecar: sc,
		LLM:     llmClient,
		Jobs:    runner,
	}

	// ---- Fiber -----------------------------------------------------------
	app := fiber.New(fiber.Config{
		BodyLimit:             50 * 1024 * 1024, // 50 MB for large scrape payloads
		DisableStartupMessage: false,
	})

	app.Use(logger.New())
	app.Use(cors.New(cors.Config{AllowOrigins: "*"}))

	// Health
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "ok", "service": "grepify-api"})
	})

	// Collections CRUD + SSE
	api := app.Group("/api")
	api.Post("/collections", deps.CreateCollection)
	api.Get("/collections", deps.ListCollections)
	api.Get("/collections/:id", deps.GetCollection)
	api.Delete("/collections/:id", deps.DeleteCollection)
	api.Post("/collections/:id/refresh", deps.RefreshCollection)
	api.Get("/collections/:id/status", deps.CollectionStatus)

	// Domains
	api.Get("/domains", deps.ListDomains)
	api.Post("/domains/:name/activate", deps.ActivateDomain)

	// Query
	api.Post("/query", deps.Query)

	// Share
	api.Get("/share/:id", deps.GetSharedAnswer)

	// ---- Start -----------------------------------------------------------
	port := getenv("PORT", "8000")
	log.Printf("Grepify API starting on :%s", port)

	// Graceful shutdown
	go func() {
		if err := app.Listen(":" + port); err != nil {
			log.Fatalf("fiber listen: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down...")
	_ = app.Shutdown()
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
