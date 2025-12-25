from unittest.mock import MagicMock, patch
from pathlib import Path

from alana_system.ingestion.text_extractor import PDFTextExtractor, PageText


def test_text_extraction():
    # Mock do pdfplumber.open para não depender de arquivo físico
    with patch("pdfplumber.open") as mock_open:
        # Mock de uma página
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Texto de exemplo da página 1"
        
        # Mock do objeto PDF retornado pelo open
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        
        mock_open.return_value = mock_pdf
        
        # Simula que o arquivo existe para passar na validação do Path.exists()
        with patch.object(Path, "exists", return_value=True):
            pdf_path = Path("data/raw/exemplo.pdf")
            extractor = PDFTextExtractor()

            pages = extractor.extract(pdf_path)

            assert isinstance(pages, list)
            assert len(pages) == 1
            
            page = pages[0]
            assert isinstance(page, PageText)
            assert page.page_number == 1
            assert page.text == "Texto de exemplo da página 1"
            assert page.char_count == len("Texto de exemplo da página 1")
