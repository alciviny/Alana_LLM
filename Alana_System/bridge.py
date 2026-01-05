"""
bridge.py

Este script atua como uma ponte (Sidecar) entre o Go e os modelos de IA Python.
Ele expõe endpoints FastAPI para embedding e geração de texto, mantendo os
modelos pré-carregados em memória para respostas de baixa latência.

Arquitetura: Senior Pattern (Sidecar / Hot-Start)
"""

import sys
from pathlib import Path
import logging

# Adiciona o diretório 'src' ao sys.path para encontrar o pacote 'alana_system'
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import CrossEncoder

try:
    from alana_system.embeddings.embedder import TextEmbedder
    from alana_system.inference.llm_engine import LLMEngine
except ImportError as e:
    logging.error(f"Erro ao importar módulos do Alana System: {e}")
    logging.error("Verifique se o 'src_path' está correto e se o ambiente virtual está ativo.")
    sys.exit(1)


# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Python Sidecar) %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURAÇÕES E INICIALIZAÇÃO DOS MODELOS (WARM START)
# =========================================================
logger.info("Iniciando o Python Sidecar para o Alana System...")

# --- Configurações ---
# Use as mesmas configurações do seu script run_search.py
MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
EMBEDDER_DEVICE = "cuda" # "cuda" para GPU, "cpu" para CPU
RERANKER_DEVICE = "cuda" # "cuda" para GPU, "cpu" para CPU
LLM_GPU_LAYERS = -1      # -1 para usar o máximo da GPU, 0 para CPU

# --- Carregamento dos Modelos ---
# Os modelos são carregados uma única vez na inicialização do servidor.
try:
    logger.info("Carregando modelo de embedding...")
    embedder = TextEmbedder(device=EMBEDDER_DEVICE)
    logger.info("✅ Modelo de embedding carregado.")
except Exception as e:
    logger.exception("❌ Falha crítica ao carregar o TextEmbedder.")
    sys.exit(1)

try:
    logger.info("Carregando modelo de Re-ranking (Cross-Encoder)...")
    # Modelo leve e rápido, ideal para reclassificação
    reranker = CrossEncoder(
        'cross-encoder/ms-marco-MiniLM-L-6-v2',
        device=RERANKER_DEVICE
    )
    logger.info("✅ Modelo de Re-ranking carregado.")
except Exception as e:
    logger.exception("❌ Falha crítica ao carregar o CrossEncoder (Re-ranker).")
    sys.exit(1)

try:
    logger.info("Carregando modelo LLM...")
    llm = LLMEngine(
        model_path=MODEL_PATH,
        n_gpu_layers=LLM_GPU_LAYERS
    )
    logger.info("✅ Modelo LLM carregado.")
except Exception as e:
    logger.exception(f"❌ Falha crítica ao carregar o LLMEngine. Verifique o caminho: {MODEL_PATH}")
    sys.exit(1)


# =========================================================
# API SERVER (FastAPI)
# =========================================================
app = FastAPI(
    title="Alana System - Python Sidecar",
    description="Servidor para realizar embedding, re-ranking e geração de texto com modelos pré-carregados.",
    version="1.1.0" # Versão atualizada
)

# --- Definição dos Schemas (Contratos da API) ---
class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    vector: list[float]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    scores: List[float]

class GenerateRequest(BaseModel):
    query: str
    context: str

class GenerateResponse(BaseModel):
    answer: str

# --- Endpoints da API ---
@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(req: EmbedRequest):
    """Gera o embedding vetorial para um texto."""
    logger.info(f"Recebido pedido de embedding para texto: '{req.text[:50]}...'")
    vector = embedder.embed_query(req.text)
    return {"vector": vector.tolist()}

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(req: RerankRequest):
    """
    Re-ranqueia uma lista de documentos com base na relevância para a query,
    usando um modelo Cross-Encoder.
    """
    logger.info(f"Recebido pedido de re-ranking para query: '{req.query[:50]}...'")
    # O Cross-Encoder espera uma lista de pares: [[query, doc1], [query, doc2], ...]
    pairs = [[req.query, doc] for doc in req.documents]
    scores = reranker.predict(pairs)
    logger.info(f"Re-ranking concluído para {len(req.documents)} documentos.")
    return {"scores": scores.tolist()}

@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(req: GenerateRequest):
    """Gera uma resposta com base em uma query e um contexto."""
    logger.info(f"Recebido pedido de geração para query: '{req.query[:50]}...'")
    answer = llm.generate_answer(query=req.query, context_text=req.context)
    return {"answer": answer}

@app.get("/health")
async def health_check():
    """Verifica se o servidor e os modelos estão operacionais."""
    # Uma verificação simples; poderia ser estendida para testar os modelos
    return {"status": "ok", "message": "Alana Sidecar está operacional."}


logger.info("🚀 Servidor FastAPI pronto para receber requisições em http://localhost:8000")

if __name__ == "__main__":
    import uvicorn
    # Isso manterá o servidor rodando e ouvindo na porta 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
