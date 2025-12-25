import sys
import os

# Adiciona o diretório 'src' ao sys.path para encontrar os módulos do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from alana_system.ingestion.pdf_loader import PDFLoader, PDFDocument



def test_pdf_discovery():
    loader = PDFLoader(raw_dir="data/raw")
    documents = loader.discover()

    assert isinstance(documents, list)
    assert len(documents) > 0

    doc = documents[0]
    assert isinstance(doc, PDFDocument)
    assert doc.id is not None
    assert doc.path.exists()
    assert doc.size_bytes > 0
