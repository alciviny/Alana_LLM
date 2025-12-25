"""
query_engine.py
Camada de Consulta Cognitiva (Query Engine)

Responsável por:
- Receber perguntas do usuário
- Gerar embedding da query
- Consultar a memória vetorial
- Retornar contexto estruturado

Não conhece:
- PDF
- Chunking
- Persistência
- Modelo LLM
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np

from src.embedding.embedder import TextEmbedder
from src.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Orquestrador de consultas semânticas.
    Atua como ponte entre usuário e memória vetorial.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
        max_context_chars: int = 4000,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_context_chars = max_context_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(self, question: str) -> Dict[str, Any]:
        """
        Executa uma consulta semântica completa.
        Retorna contexto pronto para LLM ou sumarização.
        """
        logger.info(f"Query recebida: {question}")

        query_embedding = self._embed_query(question)

        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        if not results:
            logger.warning("Nenhum contexto relevante encontrado")
            return {
                "question": question,
                "contexts": [],
                "context_text": "",
            }

        contexts = self._post_process(results)
        context_text = self._build_context_text(contexts)

        return {
            "question": question,
            "contexts": contexts,
            "context_text": context_text,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _embed_query(self, question: str) -> np.ndarray:
        """Gera embedding da query via interface do embedder."""
        return self.embedder.embed_query(question)

    def _post_process(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pós-processamento:
        - Remove duplicatas
        - Ordena por score descrescente
        """
        seen = set()
        unique_results = []

        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            cid = r.get("chunk_id")
            if cid in seen:
                continue
            seen.add(cid)
            unique_results.append(r)

        return unique_results

    def _build_context_text(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Constrói o contexto final de forma amigável para LLMs locais.
        """
        blocks: List[str] = []
        total_chars = 0

        intro = "Contexto recuperado dos documentos:\n"
        blocks.append(intro)
        total_chars += len(intro)

        for ctx in contexts:
            text = ctx.get("text", "").strip()
            if not text:
                continue

            page = ctx.get("page_number", "?")
            score = ctx.get("score", 0.0)

            header = f"### Página {page} | Relevância: {score:.2f}"
            block = f"{header}\n{text}"

            block_len = len(block)
            if total_chars + block_len > self.max_context_chars:
                logger.info("Limite de contexto atingido")
                break

            blocks.append(block)
            total_chars += block_len

        return "\n\n".join(blocks)
