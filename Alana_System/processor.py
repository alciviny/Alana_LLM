import sys
import argparse
import logging
from pathlib import Path

# Garante que o Python encontre o pacote alana_system
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from alana_system.ingestion.pdf_loader import PDFDocument
from alana_system.ingestion.audio_loader import AudioDocument
from run_ingestion import IngestionPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["PDF", "Audio", "Note"])
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    # Inicializa o pipeline (reutilizando sua lógica atual)
    pipeline = IngestionPipeline(
        raw_dir="data/raw",
        collection_name="alana_knowledge_base"
    )

    path = Path(args.path)
    
    if args.type == "PDF":
        print(f"--- Processando PDF: {path.name} ---")
        pages = pipeline.pdf_extractor.extract(path)
        pipeline._process_pages(pages, path.name, source="pdf")
    
    elif args.type == "Audio":
        print(f"--- Processando Áudio: {path.name} ---")
        pages = pipeline.audio_transcriber.transcribe(path)
        pipeline._process_pages(pages, path.name, source="audio")

    elif args.type == "Note":
        print(f"--- Processando Nota: {path.name} ---")
        pages = pipeline.note_extractor.extract(path)
        pipeline._process_pages(pages, path.name, source="note")

if __name__ == "__main__":
    main()