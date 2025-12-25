
from dataclasses import dataclass
from typing import List, Tuple
import hashlib
import logging

from ..ingestion.cleaner import CleanedPageText

logger = logging.getLogger(__name__)


# ============================================================ 
# Data Model
# ============================================================ 

@dataclass(frozen=True)
class TextChunk:
    """
    Unidade semântica mínima do sistema.
    Deve ser:
    - rastreável
    - determinística
    - semanticamente coerente
    """
    chunk_id: str
    page_number: int
    text: str
    char_count: int


# ============================================================ 
# Chunker
# ============================================================ 

class TextChunker:
    """
    Chunker semântico baseado em parágrafos,
    com janela deslizante e tolerância a dados hostis (OCR, jurídico, etc).
    """

    def __init__(
        self,
        max_chars: int = 1000,
        overlap_chars: int = 200,
        min_chars: int = 200
    ):
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars deve ser menor que max_chars")

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars

    # --------------------------------------------------------

    def chunk_pages(
        self,
        pages: List[CleanedPageText]
    ) -> List[TextChunk]:
        """
        Processa múltiplas páginas já limpas.
        """
        chunks: List[TextChunk] = []

        for page in pages:
            page_chunks = self._chunk_single_page(page)
            chunks.extend(page_chunks)

        logger.info(f"Chunking finalizado | total_chunks={len(chunks)}")
        return chunks

    # --------------------------------------------------------

    def _chunk_single_page(
        self,
        page: CleanedPageText
    ) -> List[TextChunk]:
        """
        Chunking robusto de uma única página.
        """
        paragraphs = self._split_paragraphs(page.text)
        chunks: List[TextChunk] = []

        current_paras: List[str] = []
        current_len = 0
        i = 0

        while i < len(paragraphs):
            para = paragraphs[i]
            para_len = len(para)

            # =================================================
            # CASO CRÍTICO: Parágrafo gigante (> max_chars)
            # =================================================
            if para_len > self.max_chars:
                # Se houver algo pendente no buffer, commita antes
                if current_paras:
                    self._commit_chunk(
                        chunks,
                        current_paras,
                        page.page_number
                    )
                    current_paras = []
                    current_len = 0

                logger.warning(
                    f"Parágrafo excede max_chars (chunk forçado) | "
                    f"pagina={page.page_number} | chars={para_len}"
                )

                chunks.append(
                    self._build_chunk(
                        para,
                        page.page_number
                    )
                )
                i += 1
                continue

            # =================================================
            # FLUXO NORMAL
            # =================================================
            added_len = para_len + (2 if current_paras else 0)

            # Cabe no buffer atual
            if current_len + added_len <= self.max_chars:
                current_paras.append(para)
                current_len += added_len
                i += 1
            else:
                # Chunk cheio → commit
                committed = self._commit_chunk(
                    chunks,
                    current_paras,
                    page.page_number
                )

                # Overlap semântico SÓ se o chunk foi realmente commitado
                if committed:
                    current_paras, current_len = self._build_overlap(current_paras)
                else:
                    # Se o buffer foi descartado, o overlap também é descartado.
                    # Começa do zero para evitar que o parágrafo atual se junte
                    # a um resto de buffer descartado.
                    current_paras = []
                    current_len = 0

                # NÃO incrementa i:
                # o parágrafo atual será tentado no próximo ciclo

        # =====================================================
        # Commit final
        # =====================================================
        if current_paras:
            self._commit_chunk(
                chunks,
                current_paras,
                page.page_number
            )

        return chunks

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """
        Divide texto por parágrafos lógicos.
        """
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _build_overlap(
        self,
        paragraphs: List[str]
    ) -> Tuple[List[str], int]:
        """
        Constrói overlap semântico mantendo parágrafos inteiros
        do final do chunk anterior.
        """
        overlap_paras: List[str] = []
        overlap_len = 0

        for para in reversed(paragraphs):
            para_len = len(para) + (2 if overlap_paras else 0)
            if overlap_len + para_len > self.overlap_chars:
                break
            overlap_paras.insert(0, para)
            overlap_len += para_len

        return overlap_paras, overlap_len

    def _commit_chunk(
        self,
        chunks: List[TextChunk],
        paragraphs: List[str],
        page_number: int
    ) -> bool:
        """
        Valida e adiciona chunk à lista final.
        Retorna True se o chunk foi adicionado.
        """
        if not paragraphs:
            return False

        text_block = "\n\n".join(paragraphs)

        if len(text_block) < self.min_chars:
            return False

        chunks.append(
            self._build_chunk(
                text_block,
                page_number
            )
        )
        return True

    @staticmethod
    def _build_chunk(
        text: str,
        page_number: int
    ) -> TextChunk:
        """
        Cria um chunk determinístico com hash estável.
        """
        chunk_id = hashlib.sha256(
            f"{page_number}:{text}".encode("utf-8")
        ).hexdigest()

        return TextChunk(
            chunk_id=chunk_id,
            page_number=page_number,
            text=text,
            char_count=len(text)
        )