import sys
from pathlib import Path

# Adiciona o diretório 'src' ao path
sys.path.append(str(Path(__file__).parent / "src"))

from alana_system.embeddings.embedder import TextEmbedder
from alana_system.memory.vector_store import VectorStore
from alana_system.query.query_engine import QueryEngine

def main():
    # 1. Configurar Conexão (Mesma configuração da ingestão)
    embedder = TextEmbedder(device="cpu")
    
    # IMPORTANTE: path="./qdrant_data" para ler da pasta local criada
    vector_store = VectorStore(
        collection_name="alana_knowledge_base",
        path="./qdrant_data"
    )
    
    # 2. Inicializar o Motor de Busca
    engine = QueryEngine(
        embedder=embedder,
        vector_store=vector_store,
        top_k=3  # Vai trazer os 3 trechos mais relevantes
    )

    print("\n" + "="*50)
    print("🔎 ALANA SYSTEM - BUSCA SEMÂNTICA (MODO TESTE)")
    print("="*50)

    while True:
        question = input("\nPergunte ao PDF (ou 'sair'): ")
        if question.lower() in ["sair", "exit", "quit"]:
            break

        # 3. Buscar informação
        print(f"Buscando contexto para: '{question}'...")
        result = engine.query(question)
        
        # 4. Mostrar o que ele encontrou no PDF
        if result["contexts"]:
            print(f"\n✅ Encontrei {len(result['contexts'])} trechos relevantes:\n")
            print(result["context_text"])
        else:
            print("\n❌ Não encontrei informações sobre isso no PDF.")

if __name__ == "__main__":
    main()