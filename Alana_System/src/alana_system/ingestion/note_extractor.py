import logging
from pathlib import Path
from typing import List

from .text_extractor import PageText

logger = logging.getLogger(__name__)


class NoteExtractor:
    """
    Responsável por extrair texto de arquivos de texto simples e Markdown
    de forma segura e previsível.

    Cada nota é tratada como uma única 'página lógica', mantendo
    compatibilidade com o pipeline de extração.
    """

    def extract(
        self,
        file_path: Path,
        encoding: str = "utf-8",
    ) -> List[PageText]:
        """
        Extrai o conteúdo do arquivo e retorna uma lista contendo
        um único PageText.
        """
        self._validate_file(file_path)

        try:
            logger.debug("Iniciando extração da nota: %s", file_path.name)

            content = self._read_text(file_path, encoding)
            content = self._normalize_text(content)

            if not content:
                logger.warning("Nota vazia após normalização: %s", file_path.name)

            page = PageText(
                page_number=1,
                text=content,
                char_count=len(content),
            )

            logger.info(
                "Extração concluída | %s (%d caracteres)",
                file_path.name,
                page.char_count,
            )

            return [page]

        except UnicodeDecodeError as exc:
            logger.error(
                "Erro de codificação ao ler %s (encoding=%s)",
                file_path.name,
                encoding,
            )
            raise RuntimeError(
                f"Erro de leitura: codificação incompatível em {file_path.name}"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Erro inesperado ao extrair a nota: %s",
                file_path.name,
            )
            raise RuntimeError(
                f"Falha na extração da nota: {file_path.name}"
            ) from exc

    # =====================================================
    # Internals
    # =====================================================

    @staticmethod
    def _validate_file(file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo de nota não encontrado: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Caminho informado não é um arquivo: {file_path}"
            )

    @staticmethod
    def _read_text(file_path: Path, encoding: str) -> str:
        with file_path.open("r", encoding=encoding) as file:
            return file.read()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normaliza o texto para uso posterior em pipelines de NLP/RAG.
        """
        # Remove BOM, espaços extremos e normaliza quebras de linha
        normalized = text.lstrip("\ufeff").strip()
        normalized = normalized.replace("\r\n", "\n")
        return normalized
