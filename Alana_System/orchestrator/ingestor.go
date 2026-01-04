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

	rawDir := "./Alana_System/data/raw"
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
		}

		return nil
	})
}

func processTask(workerID int, task Task) {
	fmt.Printf("[Worker %d] Processando %s: %s\n", workerID, task.Type, task.Path)

	alanaSystemDir := "Alana_System"

	// Torna o caminho do arquivo relativo ao diretório Alana_System
	relativePath, err := filepath.Rel(alanaSystemDir, task.Path)
	if err != nil {
		fmt.Printf("[Worker %d] Erro ao criar caminho relativo: %v\n", workerID, err)
		return
	}

	cmd := exec.Command(
		"python",
		"processor.py", // O script está na raiz do novo CWD
		"--type", task.Type,
		"--path", relativePath, // Passa o caminho relativo
	)
	cmd.Dir = alanaSystemDir // Define o diretório de trabalho

	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("[Worker %d] Erro: %v\n", workerID, err)
		fmt.Printf("[Worker %d] Saída do script:\n%s\n", workerID, string(output))
	} else {
		// Opcional: Imprimir a saída do script em caso de sucesso
		// fmt.Printf("[Worker %d] Saída: %s\n", workerID, string(output))
	}
}
