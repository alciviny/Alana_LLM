package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ==============================
// Domain
// ==============================

type SearchResult struct {
	Text  string
	Page  int
	Score float32
}

// Senior Pattern: Interface
type VectorSearcher interface {
	Search(ctx context.Context, vector []float32, topK uint64) ([]SearchResult, error)
}

// ==============================
// Python Sidecar Client
// ==============================

type EmbedRequest struct {
	Text string `json:"text"`
}

type EmbedResponse struct {
	Vector []float32 `json:"vector"`
}

type GenerateRequest struct {
	Query   string `json:"query"`
	Context string `json:"context"`
}

type GenerateResponse struct {
	Answer string `json:"answer"`
}

const sidecarURL = "http://localhost:8000"

// getEmbedding chama o endpoint /embed do sidecar
func getEmbedding(ctx context.Context, query string) ([]float32, error) {
	body, err := json.Marshal(EmbedRequest{Text: query})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, sidecarURL+"/embed", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embed error: %s", string(raw))
	}

	var out EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}

	return out.Vector, nil
}

// getAnswer chama o endpoint /generate do sidecar
func getAnswer(ctx context.Context, query, contextText string) (string, error) {
	body, err := json.Marshal(GenerateRequest{
		Query:   query,
		Context: contextText,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, sidecarURL+"/generate", bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("generate error: %s", string(raw))
	}

	var out GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}

	return out.Answer, nil
}

// ==============================
// Search Engine (Qdrant)
// ==============================

type AlanaEngine struct {
	client     *qdrant.Client
	collection string
	timeout    time.Duration
}

// Compile-time guarantee
var _ VectorSearcher = (*AlanaEngine)(nil)

func NewAlanaEngine(client *qdrant.Client, collection string) *AlanaEngine {
	return &AlanaEngine{
		client:     client,
		collection: collection,
		timeout:    10 * time.Second,
	}
}

// Search executa a busca vetorial REAL usando PointsClient
func (e *AlanaEngine) Search(
	ctx context.Context,
	vector []float32,
	topK uint64,
) ([]SearchResult, error) {

	ctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()

	// Cria conexão gRPC direta ao Qdrant (evita depender de métodos não expostos do cliente)
	conn, err := grpc.DialContext(ctx, "localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to dial qdrant: %w", err)
	}
	defer conn.Close()

	pointsClient := qdrant.NewPointsClient(conn)

	scoreThreshold := float32(0.3)

	resp, err := pointsClient.Search(ctx, &qdrant.SearchPoints{
		CollectionName: e.collection,
		Vector:         vector,
		Limit:          topK,
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Enable{
				Enable: true,
			},
		},
		ScoreThreshold: &scoreThreshold,
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant search failed: %w", err)
	}

	results := make([]SearchResult, 0, len(resp.GetResult()))

	for _, point := range resp.GetResult() {
		payload := point.GetPayload()

		text := ""
		if v, ok := payload["text"]; ok {
			text = v.GetStringValue()
		}

		page := 0
		if v, ok := payload["page_number"]; ok {
			page = int(v.GetIntegerValue())
		}

		results = append(results, SearchResult{
			Text:  text,
			Page:  page,
			Score: point.GetScore(),
		})
	}

	return results, nil
}

// AssembleContext monta o contexto final para o LLM
func (e *AlanaEngine) AssembleContext(
	results []SearchResult,
	tokenLimit int,
) string {

	charLimit := tokenLimit * 3

	var b strings.Builder
	b.WriteString("Contexto recuperado dos documentos:\n\n")

	for _, r := range results {
		block := fmt.Sprintf(
			"--- [Fonte/Pág %d | Score %.2f] ---\n%s\n\n",
			r.Page,
			r.Score,
			r.Text,
		)

		if b.Len()+len(block) > charLimit {
			b.WriteString("[Contexto truncado por limite de tokens]")
			break
		}

		b.WriteString(block)
	}

	return b.String()
}

// ==============================
// Main
// ==============================

func main() {
	ctx := context.Background()

	qdrantClient, err := qdrant.NewClient(&qdrant.Config{
		Host: "localhost",
		Port: 6334,
	})
	if err != nil {
		log.Fatalf("❌ Erro ao conectar no Qdrant: %v", err)
	}

	engine := NewAlanaEngine(qdrantClient, "alana_knowledge_base")

	fmt.Println("========================================")
	fmt.Println("🤖 Alana System (Go Orchestrator)")
	fmt.Println("========================================")

	question := "Qual o impacto da inteligência artificial no mercado de trabalho?"
	if len(os.Args) > 1 {
		question = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("❓ Pergunta: %s\n\n", question)

	fmt.Println("🧠 Passo 1: Gerando embedding...")
	start := time.Now()
	vector, err := getEmbedding(ctx, question)
	if err != nil {
		log.Fatalf("❌ Erro embedding: %v", err)
	}
	fmt.Printf("   OK (%v)\n\n", time.Since(start))

	fmt.Println("🔍 Passo 2: Buscando no Qdrant...")
	start = time.Now()
	results, err := engine.Search(ctx, vector, 5)
	if err != nil {
		log.Fatalf("❌ Erro busca: %v", err)
	}
	fmt.Printf("   OK (%v) | %d resultados\n\n", time.Since(start), len(results))

	fmt.Println("📝 Passo 3: Montando contexto...")
	contextText := engine.AssembleContext(results, 3000)

	fmt.Println("🤖 Passo 4: Gerando resposta...")
	start = time.Now()
	answer, err := getAnswer(ctx, question, contextText)
	if err != nil {
		log.Fatalf("❌ Erro geração: %v", err)
	}
	fmt.Printf("   OK (%v)\n\n", time.Since(start))

	fmt.Println("========================================")
	fmt.Println("✅ Resposta da Alana:")
	fmt.Println("========================================")
	fmt.Println(answer)
}
