import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# =========================================================
# PATH SETUP
# =========================================================
# Adiciona o diretório 'src' ao path para importar os módulos internos
sys.path.append(str(Path(__file__).parent / "src"))

from alana_system.embeddings.embedder import TextEmbedder
from alana_system.memory.vector_store import VectorStore
from alana_system.query.query_engine import QueryEngine
from alana_system.inference.llm_engine import LLMEngine

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================================
# TOKEN CONTROL (SIMPLIFICADO E SEGURO)
# =========================================================
def estimate_tokens(text: str) -> int:
    """
    Estimativa simples e estável:
    ~1 token ≈ 1 palavra (bom o suficiente para controle local)
    """
    if not text:
        return 0
    return len(text.split())


def truncate_context_by_budget(
    contexts: List[Dict[str, Any]], 
    max_tokens: int
) -> str:
    """
    Junta os melhores contextos respeitando o orçamento máximo de tokens.
    Extrai o texto corretamente do dicionário e adiciona metadados (página).
    """
    selected_blocks = []
    used_tokens = 0

    for item in contexts:
        # 1. Extrair o texto e página do dicionário (Correção do Bug)
        text = item.get("text", "")
        page = item.get("page_number", "?")
        
        # 2. Formatar o bloco para a IA saber a origem
        formatted_block = f"--- [Página {page}] ---\n{text}"
        
        # 3. Calcular tokens deste bloco específico
        block_tokens = estimate_tokens(formatted_block)

        # 4. Verificar se cabe no orçamento
        if used_tokens + block_tokens > max_tokens:
            logger.info(f"🛑 Orçamento atingido. Ignorando trechos restantes.")
            break

        selected_blocks.append(formatted_block)
        used_tokens += block_tokens

    logger.info(
        f"🧮 Contexto final montado: {used_tokens}/{max_tokens} tokens "
        f"({len(selected_blocks)} trechos utilizados)"
    )

    return "\n\n".join(selected_blocks)

# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 60)
    print("🤖 ALANA SYSTEM - INICIALIZAÇÃO")
    print("=" * 60)

    # -----------------------------------------------------
    # CONFIGURAÇÕES
    # -----------------------------------------------------
    # Certifique-se que este arquivo existe em 'models/'
    MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

    N_CTX = 4096  # Tamanho total da janela do modelo

    # Definição rigorosa do orçamento de tokens
    TOKEN_BUDGET = {
        "system": 300,    # Instruções do sistema
        "question": 100,  # Tamanho médio da pergunta
        "answer": 512,    # Espaço reservado para a resposta da IA
    }

    # O que sobrar é usado para o contexto dos documentos
    MAX_CONTEXT_TOKENS = (
        N_CTX
        - TOKEN_BUDGET["system"]
        - TOKEN_BUDGET["question"]
        - TOKEN_BUDGET["answer"]
    )

    logger.info(f"📐 Orçamento calculado para contexto: {MAX_CONTEXT_TOKENS} tokens")

    # -----------------------------------------------------
    # RAG COMPONENTS
    # -----------------------------------------------------
    print("📚 Inicializando memória vetorial...")
    
    # Embedder: Transforma texto em números
    embedder = TextEmbedder(device="cpu")

    # Vector Store: Banco de dados Qdrant
    vector_store = VectorStore(
        collection_name="alana_knowledge_base",
        path="./qdrant_data"
    )

    # Query Engine: Realiza a busca semântica
    query_engine = QueryEngine(
        embedder=embedder,
        vector_store=vector_store,
        top_k=5,              # Busca até 5 trechos iniciais
        score_threshold=0.35  # Filtra resultados irrelevantes
    )

    # -----------------------------------------------------
    # LLM (Cérebro)
    # -----------------------------------------------------
    print(f"🧠 Carregando LLM local: {MODEL_PATH}")
    try:
        llm = LLMEngine(
            model_path=MODEL_PATH,
            context_window=N_CTX,
            n_gpu_layers=-1  # 0 = CPU, -1 = GPU (se disponível e configurado)
        )
    except Exception as e:
        logger.error("❌ Falha crítica ao carregar modelo LLM")
        logger.error(f"Detalhe do erro: {e}")
        logger.error("DICA: Verifique se o arquivo .gguf está na pasta 'models/'")
        return

    print("\n" + "=" * 60)
    print("✅ ALANA ONLINE — Pergunte sobre seus documentos")
    print("=" * 60)

    # -----------------------------------------------------
    # LOOP DE CONVERSA
    # -----------------------------------------------------
    while True:
        try:
            question = input("\nVocê: ").strip()
        except KeyboardInterrupt:
            print("\n👋 Encerrando Alana.")
            break

        if question.lower() in {"sair", "exit", "quit"}:
            print("👋 Encerrando Alana.")
            break

        if not question:
            print("⚠️ Pergunta vazia.")
            continue

        # 1. Recuperação (Retrieval)
        logger.info("🔍 Buscando contexto relevante...")
        search_result = query_engine.query(question)
        
        # O QueryEngine retorna uma lista de dicts em 'contexts'
        raw_contexts = search_result.get("contexts", [])

        if not raw_contexts:
            print("\n❌ Alana: Não encontrei informações relevantes nos documentos para responder isso.")
            continue

        # 2. Controle de Tokens e Formatação
        #    Aqui usamos a função corrigida que lê os dicionários
        context_text = truncate_context_by_budget(
            contexts=raw_contexts,
            max_tokens=MAX_CONTEXT_TOKENS
        )

        # 3. Geração (Generation)
        logger.info("🤔 Gerando resposta...")
        try:
            answer = llm.generate_answer(
                query=question,
                context_text=context_text,
                max_tokens=TOKEN_BUDGET["answer"],
                temperature=0.1
            )

            print(f"\n🤖 Alana:\n{answer}")
            print(
                f"\n[Fonte: {len(raw_contexts)} trechos encontrados | "
                f"Contexto usado: {estimate_tokens(context_text)} tokens]"
            )

        except Exception as e:
            logger.error("❌ Erro durante inferência")
            print(f"\nErro técnico: {e}")

# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    main()