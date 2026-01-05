import logging
import time
from pathlib import Path
from typing import List
import sys

# Adiciona o diretório 'src' ao sys.path para encontrar o pacote 'alana_system'
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

from alana_system.ingestion.pdf_loader import PDFLoader
from alana_system.ingestion.text_extractor import PDFTextExtractor, PageText
from alana_system.ingestion.audio_loader import AudioLoader
from alana_system.ingestion.audio_transcriber import AudioTranscriber

from alana_system.ingestion.cleaner import TextCleaner
from alana_system.preprocessing.chunker import TextChunker
from alana_system.embeddings.embedder import TextEmbedder
from alana_system.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Pipeline unificado de ingestão multimodal (PDF + Áudio).

    Responsabilidades:
    - Orquestrar loaders
    - Padronizar PageText
    - Aplicar limpeza, chunking, embedding
    - Persistir no VectorStore
    """

    def __init__(
        self,
        raw_dir: str,
        collection_name: str,
        whisper_model: str = "small",
        embedder_device: str = "cpu",
    ):
        self.raw_dir = raw_dir

        # Loaders / Extratores
        self.pdf_loader = PDFLoader(raw_dir=raw_dir)
        self.pdf_extractor = PDFTextExtractor()

        self.audio_loader = AudioLoader(raw_dir=raw_dir)

        # Whisper é carregado UMA vez
        self.audio_transcriber = AudioTranscriber(model_size=whisper_model)

        # Pipeline comum
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(max_chars=1000, overlap_chars=200)
        self.embedder = TextEmbedder(device=embedder_device)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            host="localhost",
            port=6333
        )

    # =========================================================
    # EXECUÇÃO
    # =========================================================
    def run(self) -> None:
        start_time = time.perf_counter()
        logger.info(">>> Iniciando Pipeline de Ingestão Omni <<<")

        self._process_pdfs()
        self._process_audios()

        elapsed = time.perf_counter() - start_time
        logger.info(f">>> Pipeline concluído em {elapsed:.2f}s <<<")

    # =========================================================
    # FLUXOS
    # =========================================================
    def _process_pdfs(self) -> None:
        pdf_docs = self.pdf_loader.discover()

        for doc in pdf_docs:
            doc_start = time.perf_counter()
            try:
                logger.info(f"--- Processando PDF: {doc.name} ---")
                raw_pages = self.pdf_extractor.extract(doc.path)
                self._process_pages(raw_pages, doc.name, source="pdf")
            except Exception as e:
                logger.error(f"Erro no PDF {doc.name}: {e}")
            finally:
                elapsed = time.perf_counter() - doc_start
                logger.info(f"PDF {doc.name} finalizado em {elapsed:.2f}s")

    def _process_audios(self) -> None:
        audio_docs = self.audio_loader.discover()

        for doc in audio_docs:
            doc_start = time.perf_counter()
            try:
                logger.info(f"--- Processando Áudio: {doc.name} ---")
                raw_pages = self.audio_transcriber.transcribe(doc.path)
                self._process_pages(raw_pages, doc.name, source="audio")
            except Exception as e:
                logger.error(f"Erro no Áudio {doc.name}: {e}")
            finally:
                elapsed = time.perf_counter() - doc_start
                logger.info(f"Áudio {doc.name} finalizado em {elapsed:.2f}s")

    # =========================================================
    # PIPELINE COMUM
    # =========================================================
    def _process_pages(
        self,
        raw_pages: List[PageText],
        doc_name: str,
        source: str,
    ) -> None:
        if not raw_pages:
            logger.warning(f"Sem conteúdo extraído para {doc_name}")
            return

        # 1. Limpeza
        cleaned_pages = self.cleaner.clean_pages(raw_pages)

        # 2. Chunking
        chunks = self.chunker.chunk_pages(cleaned_pages, doc_name)

        # 3. Embedding
        embedded_chunks = self.embedder.embed_chunks(chunks)

        # 4. Persistência
        self.vector_store.upsert_embeddings(embedded_chunks)

        logger.info(
            f"{doc_name} ({source}) indexado com sucesso | "
            f"{len(chunks)} chunks"
        )


# =========================================================
# ENTRYPOINT
# =========================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    pipeline = IngestionPipeline(
        raw_dir="data/raw",
        collection_name="alana_knowledge_base",
        whisper_model="small",
        embedder_device="cpu",  # ou "cuda"
    )

    pipeline.run()


if __name__ == "__main__":
    main()
