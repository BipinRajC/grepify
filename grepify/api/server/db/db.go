// Package db manages Postgres connections and schema migrations.
package db

import (
	"context"
	"embed"
	"fmt"
	"log"

	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed schema.sql
var schemaFS embed.FS

// Pool wraps pgxpool for dependency injection.
type Pool struct {
	*pgxpool.Pool
}

// Connect creates a connection pool and runs migrations.
func Connect(ctx context.Context, dsn string) (*Pool, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse pg config: %w", err)
	}
	cfg.MaxConns = 20

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("pg connect: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("pg ping: %w", err)
	}

	log.Println("Connected to Postgres")
	return &Pool{pool}, nil
}

// Migrate runs schema.sql if tables don't exist yet.
func (p *Pool) Migrate(ctx context.Context) error {
	// Check if collections table exists
	var exists bool
	err := p.QueryRow(ctx,
		`SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'collections')`,
	).Scan(&exists)
	if err != nil {
		return fmt.Errorf("check migration: %w", err)
	}
	if exists {
		log.Println("Schema already exists, skipping migration")
		return nil
	}

	sql, err := schemaFS.ReadFile("schema.sql")
	if err != nil {
		return fmt.Errorf("read schema: %w", err)
	}

	if _, err := p.Exec(ctx, string(sql)); err != nil {
		return fmt.Errorf("run migration: %w", err)
	}

	log.Println("Schema migration applied")
	return nil
}
