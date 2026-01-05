from pathlib import Path
from dataclasses import dataclass
import hashlib
import logging
from typing import List, Iterable, Set

logger = logging.getLogger(__name__)


# =========================================================
# Domain
# =========================================================

@dataclass(frozen=True)
class NoteDocument:
    """Representa uma nota pessoal (TXT ou Markdown) no sistema."""
    id: str
    name: str
    path: Path
    size_bytes: int
    extension: str


# =========================================================
# Loader
# =========================================================

class NoteLoader:
    """
    Responsável por descobrir, validar e carregar notas de texto
    a partir de um diretório raiz.
    """

    DEFAULT_EXTENSIONS: Set[str] = {".txt", ".md"}
    DEFAULT_MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        supported_extensions: Set[str] | None = None,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    ):
        self.raw_dir = Path(raw_dir)
        self.supported_extensions = supported_extensions or self.DEFAULT_EXTENSIONS
        self.max_size_bytes = max_size_bytes

        self._validate_root_dir()

    # -----------------------------------------------------

    def discover(self) -> List[NoteDocument]:
        """
        Descobre todas as notas válidas no diretório raiz.
        Busca de forma recursiva.
        """
        documents: List[NoteDocument] = []

        for path in self._iter_candidate_files():
            try:
                documents.append(self._build_document(path))
            except Exception as exc:
                logger.warning(
                    "Nota ignorada (%s): %s",
                    path.name,
                    exc,
                    exc_info=False,
                )

        logger.info(
            "Descoberta concluída: %d notas válidas em %s",
            len(documents),
            self.raw_dir,
        )
        return documents

    # =====================================================
    # Internals
    # =====================================================

    def _validate_root_dir(self) -> None:
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Diretório de notas não encontrado: {self.raw_dir}"
            )
        if not self.raw_dir.is_dir():
            raise NotADirectoryError(
                f"Caminho informado não é um diretório: {self.raw_dir}"
            )

    def _iter_candidate_files(self) -> Iterable[Path]:
        """
        Itera apenas sobre arquivos potencialmente válidos.
        """
        for path in self.raw_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.name.startswith("."):
                logger.debug("Arquivo oculto ignorado: %s", path)
                continue

            if path.suffix.lower() not in self.supported_extensions:
                continue

            yield path

    def _build_document(self, path: Path) -> NoteDocument:
        """
        Valida e constrói o NoteDocument.
        """
        size = path.stat().st_size

        if size == 0:
            raise ValueError("Arquivo vazio")

        if size > self.max_size_bytes:
            raise ValueError(
                f"Tamanho excede limite ({size} bytes)"
            )

        doc_id = self._generate_content_hash(path)

        logger.debug("Nota carregada: %s (%d bytes)", path.name, size)

        return NoteDocument(
            id=doc_id,
            name=path.name,
            path=path.resolve(),
            size_bytes=size,
            extension=path.suffix.lower(),
        )

    # -----------------------------------------------------

    @staticmethod
    def _generate_content_hash(path: Path) -> str:
        """
        Gera hash baseado no conteúdo do arquivo para evitar duplicatas,
        mesmo em caso de renomeação.
        """
        hasher = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
