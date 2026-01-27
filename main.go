package main

import (
	"bufio"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/richinsley/jumpboot"
)

//go:embed embed.py
var embedScript string

// EmbedClient manages a Python embedding process
type EmbedClient struct {
	process *jumpboot.PythonProcess
	reader  *bufio.Reader
	writer  io.Writer
	mu      sync.Mutex
	model   string
	dim     int
}

// EmbedResponse from Python
type EmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings,omitempty"`
	Model      string      `json:"model,omitempty"`
	Dimension  int         `json:"dimension,omitempty"`
	Status     string      `json:"status,omitempty"`
	Ready      bool        `json:"ready,omitempty"`
	Error      string      `json:"error,omitempty"`
}

// NewEmbedClient creates a new embedding client with a Python process
func NewEmbedClient(envPath string, pythonVersion string, modelName string) (*EmbedClient, error) {
	// Create or use existing environment
	env, err := jumpboot.CreateEnvironmentMamba("jb-embed", envPath, pythonVersion, "conda-forge", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create environment: %w", err)
	}

	// Install dependencies if new environment
	if env.IsNew {
		fmt.Println("Installing sentence-transformers (first run, may take a minute)...")
		err = env.PipInstallPackages([]string{"sentence-transformers", "torch"}, "", "", false, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to install packages: %w", err)
		}
	}

	// Create program with embedded script
	cwd, _ := os.Getwd()
	program := &jumpboot.PythonProgram{
		Name: "jb-embed",
		Path: cwd,
		Program: jumpboot.Module{
			Name:   "__main__",
			Path:   filepath.Join(cwd, "embed.py"),
			Source: base64.StdEncoding.EncodeToString([]byte(embedScript)),
		},
	}

	// Start Python process
	process, _, err := env.NewPythonProcessFromProgram(program, nil, nil, false)
	if err != nil {
		return nil, fmt.Errorf("failed to start Python process: %w", err)
	}

	client := &EmbedClient{
		process: process,
		reader:  bufio.NewReader(process.PipeIn),
		writer:  process.PipeOut,
	}

	// Forward stderr to our stderr
	go io.Copy(os.Stderr, process.Stderr)

	// Wait for ready signal
	resp, err := client.readResponse()
	if err != nil {
		process.Terminate()
		return nil, fmt.Errorf("failed to get ready signal: %w", err)
	}
	if resp.Status != "ready" {
		process.Terminate()
		return nil, fmt.Errorf("unexpected status: %s", resp.Status)
	}

	client.model = resp.Model
	fmt.Printf("Embedding service ready (model: %s)\n", resp.Model)

	// Load specific model if requested
	if modelName != "" && modelName != resp.Model {
		if err := client.LoadModel(modelName); err != nil {
			process.Terminate()
			return nil, err
		}
	}

	return client, nil
}

func (c *EmbedClient) sendCommand(cmd map[string]interface{}) error {
	data, err := json.Marshal(cmd)
	if err != nil {
		return err
	}
	_, err = c.writer.Write(append(data, '\n'))
	return err
}

func (c *EmbedClient) readResponse() (*EmbedResponse, error) {
	line, err := c.reader.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var resp EmbedResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("python error: %s", resp.Error)
	}
	return &resp, nil
}

// LoadModel switches to a different embedding model
func (c *EmbedClient) LoadModel(name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.sendCommand(map[string]interface{}{"action": "load", "model": name}); err != nil {
		return err
	}
	resp, err := c.readResponse()
	if err != nil {
		return err
	}
	c.model = resp.Model
	c.dim = resp.Dimension
	fmt.Printf("Loaded model: %s (dimension: %d)\n", c.model, c.dim)
	return nil
}

// Embed generates embeddings for the given texts
func (c *EmbedClient) Embed(texts []string) ([][]float64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.sendCommand(map[string]interface{}{"action": "embed", "texts": texts}); err != nil {
		return nil, err
	}
	resp, err := c.readResponse()
	if err != nil {
		return nil, err
	}
	return resp.Embeddings, nil
}

// Info returns current model info
func (c *EmbedClient) Info() (*EmbedResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.sendCommand(map[string]interface{}{"action": "info"}); err != nil {
		return nil, err
	}
	return c.readResponse()
}

// Close terminates the Python process
func (c *EmbedClient) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.sendCommand(map[string]interface{}{"action": "exit"})
	c.process.Wait()
}

// --- HTTP Server ---

type Server struct {
	client *EmbedClient
	stats  struct {
		requests int64
		start    time.Time
	}
}

func (s *Server) handleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Text  string   `json:"text"`
		Texts []string `json:"texts"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	texts := req.Texts
	if req.Text != "" {
		texts = append(texts, req.Text)
	}
	if len(texts) == 0 {
		http.Error(w, "no texts provided", http.StatusBadRequest)
		return
	}

	embeddings, err := s.client.Embed(texts)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	s.stats.requests++

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"embeddings": embeddings,
		"model":      s.client.model,
		"dimension":  len(embeddings[0]),
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	info, err := s.client.Info()
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"model":     info.Model,
		"dimension": info.Dimension,
		"uptime":    time.Since(s.stats.start).String(),
		"requests":  s.stats.requests,
	})
}

func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.client.LoadModel(req.Model); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok",
		"model":  s.client.model,
	})
}

func runServer(client *EmbedClient, addr string) {
	server := &Server{client: client}
	server.stats.start = time.Now()

	http.HandleFunc("/embed", server.handleEmbed)
	http.HandleFunc("/health", server.handleHealth)
	http.HandleFunc("/model", server.handleModel)

	fmt.Printf("Starting server on %s\n", addr)
	fmt.Println("Endpoints:")
	fmt.Println("  POST /embed   - Generate embeddings")
	fmt.Println("  GET  /health  - Health check")
	fmt.Println("  POST /model   - Switch model")

	log.Fatal(http.ListenAndServe(addr, nil))
}

// --- CLI ---

func printUsage() {
	fmt.Println(`jb-embed - Local embedding service powered by sentence-transformers

Usage:
  jb-embed "text to embed"              Embed single text, output JSON
  jb-embed serve [--port PORT]          Start HTTP server (default: 8420)
  jb-embed batch                        Read texts from stdin (one per line)

Options:
  --model NAME    Model to use (default: all-MiniLM-L6-v2)
  --env PATH      Environment path (default: ~/.jb-embed/envs)
  --python VER    Python version (default: 3.11)

Examples:
  jb-embed "Hello world"
  jb-embed serve --port 8080
  echo -e "text1\ntext2" | jb-embed batch
  curl -X POST http://localhost:8420/embed -d '{"texts": ["hello", "world"]}'`)
}

func main() {
	// Defaults
	model := "all-MiniLM-L6-v2"
	envPath := filepath.Join(os.Getenv("HOME"), ".jb-embed", "envs")
	pythonVersion := "3.11"
	port := "8420"

	// Parse args
	args := os.Args[1:]
	var positional []string

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model":
			i++
			if i < len(args) {
				model = args[i]
			}
		case "--env":
			i++
			if i < len(args) {
				envPath = args[i]
			}
		case "--python":
			i++
			if i < len(args) {
				pythonVersion = args[i]
			}
		case "--port":
			i++
			if i < len(args) {
				port = args[i]
			}
		case "-h", "--help":
			printUsage()
			os.Exit(0)
		default:
			positional = append(positional, args[i])
		}
	}

	if len(positional) == 0 {
		printUsage()
		os.Exit(1)
	}

	// Create client
	client, err := NewEmbedClient(envPath, pythonVersion, model)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	cmd := positional[0]

	switch cmd {
	case "serve":
		runServer(client, ":"+port)

	case "batch":
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			text := strings.TrimSpace(scanner.Text())
			if text == "" {
				continue
			}
			embeddings, err := client.Embed([]string{text})
			if err != nil {
				log.Printf("Error: %v", err)
				continue
			}
			out, _ := json.Marshal(map[string]interface{}{
				"text":      text,
				"embedding": embeddings[0],
			})
			fmt.Println(string(out))
		}

	default:
		// Treat as text to embed
		text := strings.Join(positional, " ")
		embeddings, err := client.Embed([]string{text})
		if err != nil {
			log.Fatalf("Error: %v", err)
		}
		out, _ := json.MarshalIndent(map[string]interface{}{
			"text":      text,
			"embedding": embeddings[0],
			"dimension": len(embeddings[0]),
			"model":     client.model,
		}, "", "  ")
		fmt.Println(string(out))
	}
}
