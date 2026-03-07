package handlers

import (
	"github.com/gofiber/fiber/v2"
)

// GetSharedAnswer handles GET /api/share/:id
func (d *Deps) GetSharedAnswer(c *fiber.Ctx) error {
	shareID := c.Params("id")
	if shareID == "" {
		return c.Status(400).JSON(fiber.Map{"error": "share id required"})
	}

	answer, err := d.DB.GetSharedAnswer(c.Context(), shareID)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "not found"})
	}

	return c.JSON(answer)
}
