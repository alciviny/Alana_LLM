"""
text_extractor.py

Missão:
Extrair texto bruto de documentos PDF de forma determinística,
auditável e com máxima fidelidade possível, sem qualquer
transformação semântica.

Este módulo NÃO:
- limpa texto
- remove headers/footers
- faz chunking
- aplica OCR automaticamente
- chama modelos de IA

Ele é exclusivamente responsável por responder:
"Dado este PDF, qual texto existe em cada página?"
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    """
    Representa o texto extraído de uma única página do PDF.

    Attributes:
        page_number (int): Número da página (1-based).
        text (str): Texto bruto extraído da página.
        char_count (int): Quantidade de caracteres extraídos.
    """
    page_number: int
    text: str
    char_count: int


class PDFTextExtractor:
    """
    Extrator de texto nativo de PDFs.

    Responsabilidades:
    - Abrir PDFs
    - Iterar páginas
    - Extrair texto bruto
    - Retornar estrutura por página

    Garantias:
    - Determinístico
    - Idempotente
    - Sem efeitos colaterais
    """

    def extract(self, pdf_path: Path) -> List[PageText]:
        """
        Extrai texto bruto de todas as páginas de um PDF.

        Args:
            pdf_path (Path): Caminho do arquivo PDF.

        Returns:
            List[PageText]: Lista de objetos PageText, um por página.

        Raises:
            FileNotFoundError: Se o PDF não existir.
            RuntimeError: Se ocorrer erro na leitura do PDF.
        """

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

        pages: List[PageText] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                for index, page in enumerate(pdf.pages):
                    raw_text = page.extract_text() or ""
                    raw_text = raw_text.strip()

                    page_text = PageText(
                        page_number=index + 1,
                        text=raw_text,
                        char_count=len(raw_text)
                    )

                    pages.append(page_text)

            logger.info(
                f"Extração concluída | "
                f"arquivo={pdf_path.name} | "
                f"paginas={total_pages}"
            )

            return pages

        except Exception as exc:
            logger.exception(
                f"Falha ao extrair texto do PDF: {pdf_path.name}"
            )
            raise RuntimeError(
                f"Erro na extração de texto: {pdf_path.name}"
            ) from exc
