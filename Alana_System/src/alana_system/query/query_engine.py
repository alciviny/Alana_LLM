"""
query_engine.py
Camada de Consulta Cognitiva (Query Engine)

Responsável por:
- Receber perguntas do usuário
- Gerar embedding da query
- Consultar a memória vetorial e a memória de grafo (Busca Híbrida)
- Montar um contexto combinado para o LLM

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

from alana_system.embeddings.embedder import TextEmbedder
from alana_system.memory.vector_store import VectorStore
from alana_system.memory.graph_store import GraphStore  # Adicionado

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Orquestrador de consultas híbridas (vetorial + grafo).
    Atua como ponte entre usuário e as memórias do sistema.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        graph_store: GraphStore, # Nova dependência
        top_k: int = 5,
        score_threshold: float = 0.30,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store # Nova dependência
        self.top_k = top_k
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(self, question: str) -> Dict[str, Any]:
        """
        Executa uma consulta híbrida.
        Retorna contexto pronto para LLM.
        """
        logger.info(f"Query recebida: {question}")

        # 1. Busca Vetorial (O que você já faz)
        query_embedding = self.embedder.embed_query(question)
        vector_results = self.vector_store.search(
            query_vector=query_embedding, 
            top_k=self.top_k,
            score_threshold=self.score_threshold
        )

        # 2. Busca no Grafo (Knowledge Graph)
        # Extraímos palavras-chave simples da pergunta para buscar no SQLite
        keywords = question.split()
        graph_facts = []
        for kw in keywords:
            if len(kw) > 3: # Ignora palavras curtas
                relations = self.graph_store.query_relations(kw)
                graph_facts.extend(relations)

        # 3. Montagem do Contexto Híbrido
        context_text = self._build_hybrid_context(vector_results, graph_facts)

        return {
            "question": question,
            "context_text": context_text,
            "vector_results": vector_results,
            "graph_facts": graph_facts
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_hybrid_context(self, vector_results: List[Dict], graph_facts: List[Dict]) -> str:
        """
        Monta um prompt de contexto combinado com resultados da busca vetorial e do grafo.
        """
        context = ""

        # Monta os textos do Qdrant
        if vector_results:
            context += "### TRECHOS DE DOCUMENTOS:\n"
            for res in vector_results:
                context += f"- {res['text']}\n"
        
        # Injeta os fatos estruturados do Grafo (GraphRAG)
        if graph_facts:
            context += "\n### FATOS CONHECIDOS (RELAÇÕES):\n"
            # Limita para não poluir o contexto
            unique_facts = [dict(t) for t in {tuple(d.items()) for d in graph_facts}]
            for f in unique_facts[:10]:
                context += f"- {f['subject']} {f['relation']} {f['object']}\n"
        
        if not context:
            return "Nenhum contexto encontrado."
            
        return context
