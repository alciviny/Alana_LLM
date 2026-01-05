"""
query_engine.py
Camada de Consulta Cognitiva (Query Engine)

Responsável por:
- Receber perguntas do usuário
- Gerar embedding da query
- Consultar a memória vetorial
- Re-rankear resultados para relevância (RAG Avançado)
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
import requests

from alana_system.embeddings.embedder import TextEmbedder
from alana_system.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Orquestrador de consultas semânticas.
    Atua como ponte entre usuário e memória vetorial.
    Implementa um fluxo RAG com re-ranking.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.30,
        max_context_chars: int = 4000,
        reranker_url: str = "http://localhost:8000/rerank",
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_context_chars = max_context_chars
        self.reranker_url = reranker_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(self, question: str) -> Dict[str, Any]:
        """
        Executa uma consulta semântica completa com re-ranking.
        Retorna contexto pronto para LLM ou sumarização.
        """
        logger.info(f"Query recebida: {question}")

        # 1. Embed da query
        query_embedding = self._embed_query(question)

        # 2. Busca inicial (Funil Largo)
        # Recupera mais candidatos do que o necessário (ex: 20)
        initial_top_k = 20
        initial_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=initial_top_k,
            score_threshold=self.score_threshold,
        )

        if not initial_results:
            logger.warning("Nenhum contexto relevante encontrado na busca inicial")
            return {
                "question": question,
                "contexts": [],
                "context_text": "",
            }

        # 3. Re-ranking (Filtro de Qualidade)
        reranked_results = self._rerank_results(question, initial_results)

        # 4. Seleciona os top K finais após o re-ranking
        final_results = reranked_results[:self.top_k]

        # 5. Pós-processamento e construção do contexto
        contexts = self._post_process(final_results)
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

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Chama o serviço de re-ranking e ordena os resultados."""
        documents = [r['text'] for r in results]

        try:
            response = requests.post(
                self.reranker_url,
                json={"query": query, "documents": documents},
                timeout=10, # Adiciona um timeout
            )
            response.raise_for_status()
            scores = response.json()['scores']

            # Adiciona o novo score e ordena
            for i, res in enumerate(results):
                res['rerank_score'] = scores[i]

            # Ordena pela nova pontuação de relevância (maior é melhor)
            return sorted(results, key=lambda x: x['rerank_score'], reverse=True)

        except requests.RequestException as e:
            logger.error(f"Erro ao chamar o serviço de re-ranking: {e}")
            # Em caso de falha, retorna os resultados originais ordenados pelo score inicial
            return sorted(results, key=lambda x: x["score"], reverse=True)


    def _post_process(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pós-processamento:
        - Remove duplicatas.
        A ordenação agora é feita na etapa de re-ranking.
        """
        seen = set()
        unique_results = []

        for r in results: # A lista já vem pré-ordenada
            cid = r.get("chunk_id")
            if cid in seen:
                continue
            seen.add(cid)
            unique_results.append(r)

        return unique_results

    def _build_context_text(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Constrói o contexto final de forma amigável para LLMs locais.
        Usa o 'rerank_score' se disponível, caso contrário, o 'score' original.
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
            file_name = ctx.get("file_name", "desconhecido")
            # Prioriza o rerank_score, mas usa o score original como fallback
            relevance_score = ctx.get("rerank_score", ctx.get("score", 0.0))
            score_type = "Relevância" if "rerank_score" in ctx else "Similaridade"

            header = f"### Fonte: {file_name} | Página {page} | {score_type}: {relevance_score:.2f}"
            block = f"{header}\n{text}"

            block_len = len(block)
            if total_chars + block_len > self.max_context_chars:
                logger.info("Limite de contexto atingido")
                break

            blocks.append(block)
            total_chars += block_len

        return "\n\n".join(blocks)
