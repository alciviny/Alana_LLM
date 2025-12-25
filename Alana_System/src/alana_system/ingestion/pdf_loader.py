# src/ingestion/pdf_loader.py

from pathlib import Path
from dataclasses import dataclass
import hashlib
import logging
from typing import List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PDFDocument:
    """
    Representa um documento PDF conhecido pelo sistema.

    Missão:
    - Identidade estável
    - Metadados mínimos
    - Fonte de verdade do pipeline
    """
    id: str
    name: str
    path: Path
    size_bytes: int


class PDFLoader:
    """
    Missão:
    Descobrir, validar e registrar documentos PDF
    presentes em data/raw.

    NÃO:
    - extrai texto
    - processa conteúdo
    - executa IA
    """

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self._validate_dir()

    def _validate_dir(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.raw_dir}")

    def discover(self) -> List[PDFDocument]:
        pdfs = []

        for path in self.raw_dir.glob("*.pdf"):
            try:
                pdfs.append(self._build_document(path))
            except Exception as e:
                logger.warning(f"PDF ignorado ({path.name}): {e}")

        logger.info(f"{len(pdfs)} PDFs descobertos")
        return pdfs

    def _build_document(self, path: Path) -> PDFDocument:
        return PDFDocument(
            id=self._generate_id(path),
            name=path.name,
            path=path,
            size_bytes=path.stat().st_size
        )

    @staticmethod
    def _generate_id(path: Path) -> str:
        return hashlib.sha256(str(path).encode()).hexdigest()
