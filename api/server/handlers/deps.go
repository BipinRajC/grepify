// Package handlers contains all HTTP route handlers.
package handlers

import (
	"github.com/grepify/server/db"
	"github.com/grepify/server/jobs"
	"github.com/grepify/server/llm"
	"github.com/grepify/server/sidecar"
)

// Deps groups shared dependencies for all handlers.
type Deps struct {
	DB      *db.Pool
	Sidecar *sidecar.Client
	LLM     *llm.Client
	Jobs    *jobs.Runner
}
