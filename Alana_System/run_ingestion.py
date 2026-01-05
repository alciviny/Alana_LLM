import logging
import time
from pathlib import Path
from typing import List, Optional
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adiciona o diretório 'src' ao sys.path para encontrar o pacote 'alana_system'
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

# Loaders e Processadores de Ingestão
from alana_system.ingestion.pdf_loader import PDFLoader
from alana_system.ingestion.text_extractor import PDFTextExtractor, PageText
from alana_system.ingestion.audio_loader import AudioLoader
from alana_system.ingestion.audio_transcriber import AudioTranscriber
from alana_system.ingestion.note_loader import NoteLoader
from alana_system.ingestion.note_extractor import NoteExtractor
from alana_system.ingestion.cleaner import TextCleaner

# Componentes de IA e Memória
from alana_system.preprocessing.chunker import TextChunker
from alana_system.embeddings.embedder import TextEmbedder
from alana_system.memory.vector_store import VectorStore
from alana_system.memory.graph_store import GraphStore
from alana_system.inference.llm_engine import LLMEngine
from alana_system.preprocessing.entity_extractor import (
    EntityExtractor,
    KnowledgeGraph,
)

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Pipeline Omni: Processa PDFs, Áudios e Notas, e extrai conhecimento."""

    def __init__(
        self,
        raw_dir: str,
        collection_name: str,
        extraction_model_path: str,
        whisper_model: str = "small",
        embedder_device: str = "cpu",
    ):
        self.raw_dir = raw_dir

        # =====================================================
        # Loaders / Extratores de Conteúdo Bruto
        # =====================================================
        self.pdf_loader = PDFLoader(raw_dir=raw_dir)
        self.pdf_extractor = PDFTextExtractor()
        self.audio_loader = AudioLoader(raw_dir=raw_dir)
        self.audio_transcriber = AudioTranscriber(model_size=whisper_model)
        self.note_loader = NoteLoader(raw_dir=raw_dir)
        self.note_extractor = NoteExtractor()

        # =====================================================
        # Pipeline Comum de Processamento
        # =====================================================
        self.cleaner = TextCleaner()
        
        # --- Memória Vetorial (RAG) ---
        self.chunker = TextChunker(max_chars=800, overlap_chars=200)
        self.embedder = TextEmbedder(device=embedder_device)
        self.vector_store = VectorStore(
            collection_name=collection_name, host="localhost", port=6333
        )

        # --- Memória de Grafo (Knowledge Graph) ---
        self.graph_store = GraphStore()
        logger.info(f"Carregando LLM de extração de entidades de: {extraction_model_path}")
        self.llm_for_extraction = LLMEngine(model_path=extraction_model_path)
        self.entity_extractor = EntityExtractor(llm=self.llm_for_extraction)
        logger.info("IngestionPipeline inicializado com EntityExtractor e GraphStore")


    def run(self) -> None:
        start_time = time.perf_counter()
        logger.info(">>> Iniciando Pipeline de Ingestão Omni <<<")

        self._process_pdfs()
        self._process_audios()
        self._process_notes()

        elapsed = time.perf_counter() - start_time
        logger.info(f">>> Pipeline concluído em {elapsed:.2f}s <<<")

    # =====================================================
    # Processadores por Tipo de Fonte
    # =====================================================
    def _process_pdfs(self) -> None:
        logger.info("Iniciando ingestão de PDFs")
        pdf_docs = self.pdf_loader.discover()
        if not pdf_docs:
            logger.info("Nenhum PDF encontrado.")
            return
        for doc in pdf_docs:
            self._process_single_pdf(doc)

    def _process_single_pdf(self, doc) -> None:
        try:
            logger.info("Processando PDF | %s", doc.name)
            raw_pages = self.pdf_extractor.extract(doc.path)
            self._process_document_pages(raw_pages, doc.name, source="pdf")
        except Exception as exc:
            logger.error("Erro no PDF %s: %s", doc.name, exc)

    def _process_audios(self) -> None:
        logger.info("Iniciando ingestão de áudios")
        audio_docs = self.audio_loader.discover()
        if not audio_docs:
            logger.info("Nenhum áudio encontrado.")
            return
        for doc in audio_docs:
            self._process_single_audio(doc)

    def _process_single_audio(self, doc) -> None:
        try:
            logger.info("Processando Áudio | %s", doc.name)
            raw_pages = self.audio_transcriber.transcribe(doc.path)
            self._process_document_pages(raw_pages, doc.name, source="audio")
        except Exception as exc:
            logger.error("Erro no áudio %s: %s", doc.name, exc)
    
    def _process_notes(self) -> None:
        logger.info("Iniciando ingestão de notas pessoais")
        notes = self.note_loader.discover()
        if not notes:
            logger.info("Nenhuma nota encontrada")
            return
        for doc in notes:
            self._process_single_note(doc)

    def _process_single_note(self, doc) -> None:
        try:
            logger.info("Processando Nota | %s", doc.name)
            raw_pages = self.note_extractor.extract(doc.path)
            self._process_document_pages(raw_pages, doc.name, source="note")
        except Exception as exc:
            logger.error("Erro na nota %s: %s", doc.name, exc)

    # =====================================================
    # Lógica Central de Processamento de Documento
    # =====================================================
    def _process_document_pages(self, raw_pages: List[PageText], doc_name: str, source: str) -> None:
        """
        Lógica unificada para processar páginas.
        Agora com processamento paralelo para a extração de grafos (Entity Extractor).
        """
        if not raw_pages:
            logger.warning(f"Documento {doc_name} ({source}) não contém páginas para processar.")
            return
            
        # Limpeza inicial
        cleaned_pages = self.cleaner.clean_pages(raw_pages)

        # --- Etapa 1: Dividir em Chunks ---
        logger.info(f"Iniciando chunking para: {doc_name}")
        chunks = self.chunker.chunk_pages(cleaned_pages, doc_name)
        if not chunks:
            logger.warning(f"Nenhum chunk gerado para o documento {doc_name}.")
            return

        # --- Etapa 2: Extração Paralela de Grafo de Conhecimento ---
        logger.info(f"Iniciando extração PARALELA de entidades para {len(chunks)} chunks...")
        
        # max_workers define quantos chunks o LLM tentará processar ao mesmo tempo.
        # Se você tem uma GPU potente, 2 ou 3 é um bom número inicial.
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Criamos as tarefas para cada chunk
            futures = [
                executor.submit(
                    self._process_entities,
                    chunk.text,
                    chunk.source_name,
                    chunk.page_number
                )
                for chunk in chunks
            ]

            # Monitoramos o progresso e capturamos possíveis erros
            for future in as_completed(futures):
                try:
                    future.result() # Isso garante que exceções nas threads apareçam no log
                except Exception as exc:
                    logger.error(f"Erro na extração paralela de um chunk em {doc_name}: {exc}")

        # --- Etapa 3: Indexação Vetorial (RAG) ---
        logger.info(f"Iniciando indexação vetorial para {len(chunks)} chunks...")
        embedded_chunks = self.embedder.embed_chunks(chunks)
        self.vector_store.upsert_embeddings(embedded_chunks)
        
        logger.info(f"'{doc_name}' ({source}) concluído com sucesso.")

    def _process_entities(self, text: str, doc_name: str, page_number: int) -> None:
        """
        Extrai conhecimento e persiste no banco de dados SQLite.
        """
        if not text.strip():
            return

        # 1. Extração semântica via LLM
        # Aqui o LLM trabalha pesado
        graph = self.entity_extractor.extract_graph(text)

        if graph.entities or graph.relations:
            # 2. Persistência no SQLite
            self.graph_store.add_knowledge(
                graph=graph,
                source_doc=doc_name,
                page_number=page_number
            )
            logger.info(f"Conhecimento persistido no Grafo | doc={doc_name} page={page_number}")


# =========================================================
# ENTRYPOINT
# =========================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # Configuração centralizada
    RAW_DATA_DIR = "data/raw"
    KNOWLEDGE_BASE_NAME = "alana_knowledge_base"
    WHISPER_MODEL = "small"
    EMBEDDER_DEVICE = "cuda"  # ou "cuda"
    EXTRACTION_MODEL = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

    # Verifica se o diretório de dados brutos existe
    if not Path(RAW_DATA_DIR).is_dir():
        logger.error(f"Diretório de dados brutos '{RAW_DATA_DIR}' não encontrado. Crie-o e adicione seus arquivos lá.")
        sys.exit(1)
        
    # Verifica se o modelo de extração existe
    if not Path(EXTRACTION_MODEL).is_file():
        logger.error(f"Modelo de extração '{EXTRACTION_MODEL}' não encontrado. Verifique o caminho.")
        # Opcional: Adicionar link/instrução de como baixar o modelo.
        sys.exit(1)

    pipeline = IngestionPipeline(
        raw_dir=RAW_DATA_DIR,
        collection_name=KNOWLEDGE_BASE_NAME,
        extraction_model_path=EXTRACTION_MODEL,
        whisper_model=WHISPER_MODEL,
        embedder_device=EMBEDDER_DEVICE,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
