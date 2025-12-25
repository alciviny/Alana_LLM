"""
vector_store.py
Camada de Memória Vetorial (Vector Store)

Responsável por persistir, versionar e consultar embeddings
Backend: Qdrant (local/offline via Docker)

GARANTIAS DE SEGURANÇA:
- Conversão determinística e válida de SHA-256 -> UUID (uuid5)
- Validação defensiva da dimensão do embedding
- Upsert em batches para evitar timeouts
- Preservação do ID original no payload
- Indexação explícita de campos do payload (performance em filtros)
"""

from __future__ import annotations

import uuid
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
)

from ..embeddings.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Gerenciador de persistência vetorial (Qdrant).

    Responsabilidades:
    - Criar e validar collections
    - Persistir embeddings de forma idempotente
    - Executar busca semântica eficiente

    NÃO conhece:
    - PDF
    - Chunker
    - LLM
    - Prompt
    """

    def __init__(
        self,
        collection_name: str,
        host: Optional[str] = None,
        port: int = 6333,
        location: Optional[str] = None,
        path: Optional[str] = None,
        vector_dim: int = 384,
        distance: Distance = Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.distance = distance

        if location or path:
            self.client = QdrantClient(location=location, path=path)
        else:
            self.client = QdrantClient(host=host or "localhost", port=port)

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            logger.info(f"Collection já existe: {self.collection_name}")
            return

        logger.info(
            f"Criando collection '{self.collection_name}' | dim={self.vector_dim}"
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=self.distance,
            ),
        )

    # ------------------------------------------------------------------
    # Payload Index
    # ------------------------------------------------------------------
    def create_payload_index(
        self,
        field_name: str,
        field_type: str = "integer",
    ) -> None:
        """
        Cria um índice em um campo do payload para acelerar filtros.

        Exemplos:
        - page_number -> integer
        - original_id -> keyword
        """
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            logger.info(
                f"Índice de payload criado | "
                f"campo='{field_name}' tipo='{field_type}'"
            )
        except Exception as e:
            # O Qdrant lança erro se o índice já existir
            logger.warning(
                f"Falha ao criar índice de payload "
                f"(talvez já exista) | campo='{field_name}' | erro={e}"
            )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    def upsert_embeddings(
        self,
        chunks: List[EmbeddedChunk],
        batch_size: int = 100,
    ) -> None:
        """
        Insere embeddings em batches.
        Garante:
        - UUID válido e determinístico
        - Dimensão correta do vetor
        """
        if not chunks:
            logger.warning("Nenhum embedding para inserir")
            return

        total = len(chunks)
        logger.info(f"Iniciando upsert de {total} vetores")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            points: List[PointStruct] = []

            for chunk in batch:
                # -------------------------------
                # Validação defensiva do embedding
                # -------------------------------
                if len(chunk.embedding) != self.vector_dim:
                    raise ValueError(
                        f"Dimensão inválida do embedding | "
                        f"esperado={self.vector_dim} "
                        f"recebido={len(chunk.embedding)} "
                        f"chunk_id={chunk.chunk_id}"
                    )

                # -------------------------------
                # Conversão determinística SHA -> UUID
                # -------------------------------
                uuid_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        chunk.chunk_id
                    )
                )

                payload = {
                    "original_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                }

                points.append(
                    PointStruct(
                        id=uuid_id,
                        vector=chunk.embedding.tolist(),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.debug(
                f"Upsert batch concluído | "
                f"range={i}-{i + len(batch)}"
            )

        logger.info("Upsert finalizado com sucesso")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Filter] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Busca semântica por similaridade vetorial.
        Retorna texto, score e metadados.
        """
        if len(query_vector) != self.vector_dim:
            raise ValueError(
                f"Dimensão inválida do query_vector | "
                f"esperado={self.vector_dim} "
                f"recebido={len(query_vector)}"
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=filters,
            score_threshold=score_threshold,
        )

        response: List[Dict[str, Any]] = []
        for r in results:
            response.append(
                {
                    "score": r.score,
                    "chunk_id": r.payload.get("original_id"),
                    "page_number": r.payload.get("page_number"),
                    "text": r.payload.get("text"),
                }
            )

        return response

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Retorna o número de vetores na collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
