
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
    source_name: str


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
        pages: List[CleanedPageText],
        source_name: str
    ) -> List[TextChunk]:
        """
        Processa múltiplas páginas já limpas.
        """
        chunks: List[TextChunk] = []

        for page in pages:
            page_chunks = self._chunk_single_page(page, source_name)
            chunks.extend(page_chunks)

        logger.info(f"Chunking finalizado | total_chunks={len(chunks)}")
        return chunks

    # --------------------------------------------------------

    def _chunk_single_page(
        self,
        page: CleanedPageText,
        source_name: str
    ) -> List[TextChunk]:
        """
        Chunking robusto com suporte a textos contínuos (Audio/OCR).
        """
        # 1. Tenta dividir por parágrafos naturais
        paragraphs = self._split_paragraphs(page.text)
        chunks: List[TextChunk] = []

        current_paras: List[str] = []
        current_len = 0
        i = 0

        while i < len(paragraphs):
            para = paragraphs[i]
            para_len = len(para)

            # =================================================
            # CASO CRÍTICO: Parágrafo gigante (Whisper/OCR ruim)
            # =================================================
            if para_len > self.max_chars:
                # Se tem buffer pendente, salva ele antes
                if current_paras:
                    self._commit_chunk(chunks, current_paras, page.page_number, source_name)
                    current_paras = []
                    current_len = 0

                logger.info(
                    f"🔪 Fatiando bloco gigante | pagina={page.page_number} | chars={para_len}"
                )
                
                # --- NOVA LÓGICA: FATIAMENTO FORÇADO ---
                # Divide o textão em fatias menores que respeitam max_chars
                sub_blocks = self._split_text_by_limit(para, self.max_chars, self.overlap_chars)
                
                for block in sub_blocks:
                    chunks.append(self._build_chunk(block, page.page_number, source_name))
                
                i += 1
                continue

            # =================================================
            # FLUXO NORMAL (mantido igual)
            # =================================================
            added_len = para_len + (2 if current_paras else 0)

            if current_len + added_len <= self.max_chars:
                current_paras.append(para)
                current_len += added_len
                i += 1
            else:
                committed = self._commit_chunk(chunks, current_paras, page.page_number, source_name)
                if committed:
                    current_paras, current_len = self._build_overlap(current_paras)
                else:
                    current_paras = []
                    current_len = 0

        # Commit final
        if current_paras:
            self._commit_chunk(chunks, current_paras, page.page_number, source_name)

        return chunks

    # --------------------------------------------------------
    # NOVO MÉTODO AUXILIAR
    # --------------------------------------------------------
    def _split_text_by_limit(self, text: str, limit: int, overlap: int) -> List[str]:
        """
        Fatia um texto contínuo em blocos de tamanho fixo com overlap.
        Versão corrigida para evitar loop infinito em trechos finais curtos.
        """
        blocks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # Define o fim ideal
            end = min(start + limit, text_len)
            
            # Ajuste inteligente: tenta não cortar no meio da palavra
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space

            # Extrai o bloco
            block = text[start:end].strip()
            if block:
                blocks.append(block)
            
            # --- CORREÇÃO DO LOOP INFINITO ---
            # Calcula o próximo início recuando pelo overlap
            next_start = end - overlap

            # Se o recuo fizer a gente ficar no mesmo lugar (ou voltar), 
            # forçamos o avanço para o fim do bloco atual.
            # Isso acontece quando o bloco é menor que o overlap (final do texto).
            if next_start <= start:
                next_start = end
            
            start = next_start

        return blocks

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
        page_number: int,
        source_name: str
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
                page_number,
                source_name
            )
        )
        return True

    @staticmethod
    def _build_chunk(
        text: str,
        page_number: int,
        source_name: str
    ) -> TextChunk:
        """
        Cria um chunk determinístico com hash estável.
        """
        chunk_id = hashlib.sha256(
            f"{source_name}:{page_number}:{text}".encode("utf-8")
        ).hexdigest()

        return TextChunk(
            chunk_id=chunk_id,
            page_number=page_number,
            text=text,
            char_count=len(text),
            source_name=source_name
        )