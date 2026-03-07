package handlers

import (
	"log"

	"github.com/gofiber/fiber/v2"
)

// ListDomains handles GET /api/domains
func (d *Deps) ListDomains(c *fiber.Ctx) error {
	domains, err := d.DB.ListDomains(c.Context())
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(domains)
}

// ActivateDomain handles POST /api/domains/:name/activate
// Creates a collection from a pre-defined domain preset.
func (d *Deps) ActivateDomain(c *fiber.Ctx) error {
	name := c.Params("name")

	domain, err := d.DB.GetDomain(c.Context(), name)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "domain not found"})
	}

	var body struct {
		Limit int `json:"limit"`
	}
	_ = c.BodyParser(&body)
	if body.Limit == 0 {
		body.Limit = 50
	}

	col, err := d.DB.CreateCollection(c.Context(), domain.Label, domain.Description, domain.Name, domain.Subreddits)
	if err != nil {
		log.Printf("activate domain error: %v", err)
		return c.Status(500).JSON(fiber.Map{"error": "failed to create collection"})
	}

	jobID, err := d.DB.CreateJob(c.Context(), col.ID, "index")
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	d.Jobs.Enqueue(col.ID, jobID, domain.Subreddits, body.Limit)

	return c.Status(201).JSON(fiber.Map{
		"collection": col,
		"job_id":     jobID,
		"domain":     domain,
		"message":    "Domain collection created and indexing started",
	})
}
