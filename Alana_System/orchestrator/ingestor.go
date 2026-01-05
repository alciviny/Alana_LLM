package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
)

type Task struct {
	Path string
	Type string
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Captura Ctrl+C
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sig
		fmt.Println("\n⛔ Cancelando ingestão...")
		cancel()
	}()

	// AJUSTE: Caminho relativo para quem está dentro de Alana_System
	rawDir := "./data/raw" 
	numWorkers := 4

	tasks := make(chan Task, 100)
	var wg sync.WaitGroup

	// Workers
	for i := 1; i <= numWorkers; i++ {
		wg.Add(1)
		go worker(ctx, i, tasks, &wg)
	}

	// Descoberta de arquivos
	if err := discoverFiles(ctx, rawDir, tasks); err != nil {
		fmt.Println("Erro na descoberta:", err)
	}

	close(tasks)
	wg.Wait()

	fmt.Println("✅ Ingestão concluída pelo Orquestrador Go")
}

func worker(ctx context.Context, id int, tasks <-chan Task, wg *sync.WaitGroup) {
	defer wg.Done()

	for {
		select {
		case <-ctx.Done():
			fmt.Printf("[Worker %d] Cancelado\n", id)
			return
		case task, ok := <-tasks:
			if !ok {
				return
			}
			processTask(id, task)
		}
	}
}

func discoverFiles(ctx context.Context, root string, tasks chan<- Task) error {
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		switch filepath.Ext(path) {
		case ".pdf":
			tasks <- Task{Path: path, Type: "PDF"}
		case ".mp3", ".wav", ".m4a":
			tasks <- Task{Path: path, Type: "Audio"}
		case ".txt", ".md":
			tasks <- Task{Path: path, Type: "Note"}
		}

		return nil
	})
}

func processTask(workerID int, task Task) {
	fmt.Printf("[Worker %d] Processando %s: %s\n", workerID, task.Type, task.Path)

	// AJUSTE: O diretório de trabalho agora é o atual (.)
	alanaSystemDir := "." 

	// Torna o caminho do arquivo relativo ao diretório atual
	relativePath, err := filepath.Rel(alanaSystemDir, task.Path)
	if err != nil {
		fmt.Printf("[Worker %d] Erro ao criar caminho relativo: %v\n", workerID, err)
		return
	}

	cmd := exec.Command(
		"python",
		"processor.py", 
		"--type", task.Type,
		"--path", relativePath,
	)
	cmd.Dir = alanaSystemDir 

	output, err := cmd.CombinedOutput()
	
	// AJUSTE: Mostrar sempre a saída do Python para debug (ajuda a ver o progresso do Whisper)
	if len(output) > 0 {
		fmt.Printf("[Worker %d] Saída do Python:\n%s\n", workerID, string(output))
	}

	if err != nil {
		fmt.Printf("[Worker %d] Erro crítico no Worker: %v\n", workerID, err)
	}
}