"""
embedder.py

Missão:
Converter TextChunk em embeddings vetoriais densos,
de forma offline, escalável, multilíngue e segura para produção.

Este módulo NÃO:
- faz busca vetorial
- chama LLM
- constrói prompts
"""

from dataclasses import dataclass
from typing import List, Generator
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch
except ImportError:  # torch é opcional
    torch = None

from ..preprocessing.chunker import TextChunk

logger = logging.getLogger(__name__)


# ============================================================
# Data Model
# ============================================================

@dataclass(frozen=True)
class EmbeddedChunk:
    """
    Chunk com embedding associado.
    """
    chunk_id: str
    page_number: int
    text: str
    embedding: np.ndarray


# ============================================================
# Embedder
# ============================================================

class TextEmbedder:
    """
    Embedder offline, multilíngue e memory-safe.

    Projetado para:
    - grandes volumes de documentos
    - execução 100% offline
    - integração posterior com FAISS / HNSW
    """

    def __init__(
        self,
        # Default ajustado para português / multilíngue
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 32,
        normalize: bool = True,
        device: str | None = None,
    ):
        self.batch_size = batch_size
        self.normalize = normalize

        # Auto-detecção de device
        if device is None:
            if torch and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info(
            f"Carregando modelo de embedding | "
            f"modelo={model_name} | device={device}"
        )

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

    # --------------------------------------------------------

    def embed_chunks(
        self,
        chunks: List[TextChunk]
    ) -> List[EmbeddedChunk]:
        """
        Converte TextChunk em EmbeddedChunk de forma incremental,
        evitando picos de memória.
        """
        embedded_chunks: List[EmbeddedChunk] = []

        total = len(chunks)
        processed = 0

        for batch in self._batch_generator(chunks, self.batch_size):
            batch_texts = [c.text for c in batch]

            embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )

            for chunk, emb in zip(batch, embeddings):
                embedded_chunks.append(
                    EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        text=chunk.text,
                        embedding=emb,
                    )
                )

            processed += len(batch)
            logger.debug(
                f"Embeddings processados | {processed}/{total}"
            )

        if embedded_chunks:
            dim = embedded_chunks[0].embedding.shape[0]
        else:
            dim = 0

        logger.info(
            f"Embedding concluído | total={len(embedded_chunks)} | dim={dim}"
        )

        return embedded_chunks

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _batch_generator(
        data: List[TextChunk],
        size: int
    ) -> Generator[List[TextChunk], None, None]:
        """
        Generator de batches para evitar carregar
        grandes volumes de dados em memória.
        """
        for i in range(0, len(data), size):
            yield data[i : i + size]
