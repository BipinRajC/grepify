// Package llm provides LLM clients (Groq, extensible to others).
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

var httpClient = &http.Client{Timeout: 60 * time.Second}

// Client wraps an OpenAI-compatible chat API (Groq, OpenAI, etc.)
type Client struct {
	APIKey  string
	Model   string
	BaseURL string // default: Groq
}

func NewGroq(apiKey, model string) *Client {
	if model == "" {
		model = "llama-3.3-70b-versatile"
	}
	return &Client{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: "https://api.groq.com/openai/v1/chat/completions",
	}
}

// ChatMessage is an OpenAI-compatible message.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// Complete sends a chat completion request and returns the response text.
func (c *Client) Complete(system, user string) (string, error) {
	if c.APIKey == "" {
		return "[LLM unavailable — set GROQ_API_KEY]", nil
	}

	msgs := []ChatMessage{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}

	return c.complete(msgs, 0.3, 1024)
}

// CompleteJSON sends a prompt expecting JSON output.
func (c *Client) CompleteJSON(system, user string) (string, error) {
	if c.APIKey == "" {
		return "{}", nil
	}
	msgs := []ChatMessage{
		{Role: "system", Content: system + "\nRespond ONLY with valid JSON, no markdown."},
		{Role: "user", Content: user},
	}
	return c.complete(msgs, 0.2, 2048)
}

func (c *Client) complete(messages []ChatMessage, temp float64, maxTokens int) (string, error) {
	req := chatRequest{
		Model:       c.Model,
		Messages:    messages,
		Temperature: temp,
		MaxTokens:   maxTokens,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}

	httpReq, err := http.NewRequest("POST", c.BaseURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("llm request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("llm HTTP %d: %s", resp.StatusCode, string(b))
	}

	var chatResp chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", fmt.Errorf("llm decode: %w", err)
	}
	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("llm empty response")
	}
	return chatResp.Choices[0].Message.Content, nil
}
