"""
cleaner.py

Missão:
Normalizar texto bruto extraído de documentos, removendo ruídos
técnicos comuns (quebras excessivas, espaços, artefatos),
sem alterar significado, contexto ou estrutura lógica.

Este módulo NÃO:
- resume
- reescreve
- corrige semântica
- aplica chunking
- usa modelos de IA
"""

from dataclasses import dataclass
from typing import List
import logging
import re

from .text_extractor import PageText

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanedPageText:
    """
    Representa o texto de uma página após limpeza controlada.

    Attributes:
        page_number (int): Número da página.
        text (str): Texto limpo.
        original_char_count (int): Caracteres antes da limpeza.
        cleaned_char_count (int): Caracteres após limpeza.
    """
    page_number: int
    text: str
    original_char_count: int
    cleaned_char_count: int


class TextCleaner:
    """
    Normalizador de texto bruto.

    Filosofia:
    - regras simples
    - efeitos previsíveis
    - nenhuma inteligência implícita
    """

    def clean_pages(self, pages: List[PageText]) -> List[CleanedPageText]:
        cleaned_pages: List[CleanedPageText] = []

        for page in pages:
            cleaned_text = self._clean_text(page.text)

            cleaned_pages.append(
                CleanedPageText(
                    page_number=page.page_number,
                    text=cleaned_text,
                    original_char_count=page.char_count,
                    cleaned_char_count=len(cleaned_text)
                )
            )

        logger.info(
            f"Limpeza concluída | paginas={len(cleaned_pages)}"
        )

        return cleaned_pages

    def _clean_text(self, text: str) -> str:
        """
        Pipeline de limpeza controlada.

        Ordem importa.
        """

        if not text:
            return ""

        text = self._normalize_whitespace(text)
        text = self._remove_hyphenation(text)
        text = self._fix_line_breaks(text)

        return text.strip()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        # Converte múltiplos espaços em um
        text = re.sub(r"[ \t]+", " ", text)
        # Normaliza múltiplas quebras de linha
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _remove_hyphenation(text: str) -> str:
        """
        Remove hifenização comum de quebra de linha:
        ex: 'informa-\nção' → 'informação'
        """
        return re.sub(r"-\n(\w)", r"\1", text)

    @staticmethod
    def _fix_line_breaks(text: str) -> str:
        """
        Une linhas quebradas artificialmente mantendo parágrafos.
        """
        lines = text.split("\n")
        fixed_lines = []

        buffer = ""

        for line in lines:
            line = line.strip()

            if not line:
                if buffer:
                    fixed_lines.append(buffer)
                    buffer = ""
                fixed_lines.append("")
            else:
                if buffer:
                    buffer += " " + line
                else:
                    buffer = line

        if buffer:
            fixed_lines.append(buffer)

        return "\n".join(fixed_lines)
