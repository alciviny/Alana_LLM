from pathlib import Path
from dataclasses import dataclass
import hashlib
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioDocument:
    """
    Representa um arquivo de áudio descoberto pelo sistema.

    Este objeto é puramente descritivo:
    - NÃO transcreve
    - NÃO normaliza
    - NÃO chama modelos

    Ele apenas carrega metadados confiáveis para o pipeline de ingestão.
    """
    id: str
    name: str
    path: Path
    size_bytes: int
    format: str
    duration_seconds: Optional[float] = None


class AudioLoader:
    """
    Missão:
    Descobrir e validar arquivos de áudio no diretório data/raw.

    Formatos suportados:
    mp3, wav, m4a, ogg, flac

    Responsabilidade única:
    - Descoberta
    - Validação
    - Criação de AudioDocument
    """

    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self._validate_dir()

    def _validate_dir(self) -> None:
        if not self.raw_dir.exists() or not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Diretório de áudio não encontrado: {self.raw_dir}")

    def discover(self) -> List[AudioDocument]:
        """
        Descobre recursivamente arquivos de áudio válidos.
        """
        audio_docs: List[AudioDocument] = []

        for path in self.raw_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    audio_docs.append(self._build_document(path))
                except Exception as e:
                    logger.warning(f"Áudio ignorado ({path.name}): {e}")

        logger.info(f"{len(audio_docs)} arquivos de áudio descobertos em {self.raw_dir}")
        return audio_docs

    def _build_document(self, path: Path) -> AudioDocument:
        return AudioDocument(
            id=self._generate_content_hash(path),
            name=path.name,
            path=path.resolve(),
            size_bytes=path.stat().st_size,
            format=path.suffix.lower(),
            duration_seconds=None  # preenchido futuramente pelo parser
        )

    @staticmethod
    def _generate_content_hash(path: Path, chunk_size: int = 8192) -> str:
        """
        Gera um hash SHA-256 baseado no conteúdo do arquivo.
        Mais estável que hash baseado no path.
        """
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
