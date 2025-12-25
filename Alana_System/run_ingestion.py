import logging
import sys
from pathlib import Path

# Ajusta o path para encontrar os módulos dentro de src/
sys.path.append(str(Path(__file__).parent / "src"))

from alana_system.ingestion.pdf_loader import PDFLoader
from alana_system.ingestion.text_extractor import PDFTextExtractor
from alana_system.ingestion.cleaner import TextCleaner
from alana_system.preprocessing.chunker import TextChunker
from alana_system.embeddings.embedder import TextEmbedder
from alana_system.memory.vector_store import VectorStore

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    logger.info(">>> Iniciando Pipeline de Ingestão Alana System <<<")

    # 1. Configurações
    RAW_DIR = "data/raw"
    COLLECTION_NAME = "alana_knowledge_base"
    
    # 2. Inicializar Componentes
    loader = PDFLoader(raw_dir=RAW_DIR)
    extractor = PDFTextExtractor()
    cleaner = TextCleaner()
    chunker = TextChunker(max_chars=1000, overlap_chars=200)

    # ✅ Forçar uso da GPU se disponível
    embedder = TextEmbedder(device="cpu")  

   
    # O parametro 'path' diz ao sistema para salvar numa pasta local, sem usar Docker
    vector_store = VectorStore(
        collection_name=COLLECTION_NAME,
        path="./qdrant_data")
        # 3. Descobrir PDFs
    documents = loader.discover()
    if not documents:
        logger.warning("Nenhum PDF encontrado em data/raw/. Adicione arquivos para processar.")
        return

    # 4. Processar cada documento
    for doc in documents:
        try:
            logger.info(f"--- Processando: {doc.name} ---")
            
            # A. Extração
            raw_pages = extractor.extract(doc.path)
            logger.info(f"Extração concluída | {len(raw_pages)} páginas")

            # B. Limpeza
            cleaned_pages = cleaner.clean_pages(raw_pages)
            logger.info("Limpeza de texto concluída")

            # C. Chunking
            chunks = chunker.chunk_pages(cleaned_pages)
            logger.info(f"Chunking concluído | {len(chunks)} chunks gerados")

            # D. Embedding (GPU)
            embedded_chunks = embedder.embed_chunks(chunks)
            logger.info("Embeddings gerados com sucesso (GPU)")

            # E. Persistência (Salva no Qdrant)
            vector_store.upsert_embeddings(embedded_chunks)
            logger.info(f"{doc.name} indexado na memória com sucesso.")

        except Exception as e:
            logger.error(f"Erro ao processar {doc.name}: {e}", exc_info=True)

    logger.info(">>> Ingestão Concluída com Sucesso! <<<")

if __name__ == "__main__":
    main()
