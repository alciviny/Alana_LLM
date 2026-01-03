"""
tests/test_chunker.py

Testes para a lógica de chunking semântico.
"""
import pytest
from alana_system.preprocessing.chunker import TextChunker
from alana_system.ingestion.cleaner import CleanedPageText

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def chunker():
    """Retorna uma instância do TextChunker com parâmetros padrão para teste."""
    return TextChunker(max_chars=100, overlap_chars=30, min_chars=20)

@pytest.fixture
def single_page():
    """Retorna uma página de texto limpo para os testes."""
    text = (
        "Primeiro parágrafo do texto. Ele é curto e direto.\n\n"
        "Segundo parágrafo é um pouco mais longo, com mais detalhes. "
        "Ele serve para testar a junção de parágrafos em um único chunk.\n\n"
        "Terceiro. Curto.\n\n"
        "Quarto parágrafo é deliberadamente muito, muito, muito longo para exceder o limite "
        "de caracteres de cem, forçando o chunker a tratá-lo como um caso especial e isolado."
    )
    return CleanedPageText(
        page_number=1,
        text=text,
        original_char_count=len(text),
        cleaned_char_count=len(text),
    )

# ============================================================
# Test Cases
# ============================================================

def test_chunking_normal(chunker, single_page):
    """
    Testa o fluxo normal de chunking, onde parágrafos são agrupados
    respeitando o `max_chars`.
    """
    chunks = chunker.chunk_pages([single_page])

    # Rastreamento da lógica do chunker:
    # paragraphs = ["p1" (46), "p2" (128), "p3" (18), "p4" (>100)]
    # 1. i=0, para=p1. Adiciona ao buffer. current_paras=["p1"], current_len=46. i++.
    # 2. i=1, para=p2 (len > max_chars).
    #    - Buffer não está vazio, commita buffer: _commit_chunk(["p1"]). len > min_chars. CHUNK 1 criado.
    #    - Buffer esvaziado.
    #    - Commita parágrafo gigante: _build_chunk(p2). CHUNK 2 criado.
    #    - i++.
    # 3. i=2, para=p3. Adiciona ao buffer. current_paras=["p3"], current_len=18. i++.
    # 4. i=3, para=p4 (len > max_chars).
    #    - Buffer não está vazio, commita buffer: _commit_chunk(["p3"]). len < min_chars. NENHUM CHUNK criado.
    #    - Buffer esvaziado.
    #    - Commita parágrafo gigante: _build_chunk(p4). CHUNK 3 criado.
    #    - i++.
    # Fim do loop.
    
    assert len(chunks) == 3
    assert "Primeiro parágrafo" in chunks[0].text
    assert "Segundo parágrafo" in chunks[1].text
    assert "Quarto parágrafo" in chunks[2].text

def test_paragraph_too_long(chunker):
    """
    Testa o caso crítico onde um único parágrafo excede `max_chars`.
    Ele deve se tornar um chunk sozinho, mesmo que muito grande.
    """
    long_paragraph = "a" * 150
    page = CleanedPageText(
        page_number=1,
        text=long_paragraph,
        original_char_count=len(long_paragraph),
        cleaned_char_count=len(long_paragraph),
    )
    
    chunks = chunker.chunk_pages([page])
    
    assert len(chunks) == 1
    assert chunks[0].text == long_paragraph
    assert chunks[0].char_count == 150

def test_overlap_logic(chunker):
    """
    Testa se o overlap semântico está funcionando, mantendo parágrafos.
    """
    # p1(30) + p2(30) + p3(30) -> len=90 + 4 (separadores) = 94. Cabe no chunk.
    # p4(30) não cabe (94 + 2 + 30 > 100). Chunk [p1,p2,p3] é commitado.
    # Overlap é construído a partir do fim, com limite de 30 chars. p3 (30) entra.
    # Próximo chunk começa com o overlap [p3] e adiciona p4.
    p1 = "a" * 30
    p2 = "b" * 30
    p3 = "c" * 30
    p4 = "d" * 30
    text = f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}"
    page = CleanedPageText(
        page_number=1,
        text=text,
        original_char_count=len(text),
        cleaned_char_count=len(text),
    )

    chunks = chunker.chunk_pages([page])

    assert len(chunks) == 2
    assert chunks[0].text == f"{p1}\n\n{p2}\n\n{p3}"
    assert chunks[1].text == f"{p3}\n\n{p4}"

def test_min_chars_logic(chunker):
    """
    Testa se parágrafos ou chunks combinados que não atingem `min_chars`
    são descartados.
    """
    p1 = "a" * 15  # < min_chars
    p2 = "b" * 50
    text = f"{p1}\n\n{p2}"
    page = CleanedPageText(
        page_number=1,
        text=text,
        original_char_count=len(text),
        cleaned_char_count=len(text),
    )

    chunks = chunker.chunk_pages([page])
    
    # Rastreamento:
    # 1. i=0, para=p1(15). Adiciona ao buffer. current_len=15.
    # 2. i=1, para=p2(50). 15 + 2 + 50 <= 100. Cabe no buffer. current_len=67.
    # Fim da página. Commita o buffer [p1, p2]. len > min_chars. CHUNK 1 criado.
    assert len(chunks) == 1
    assert chunks[0].text == f"{p1}\n\n{p2}"

def test_chunk_id_is_deterministic(chunker):
    """
    Garante que o mesmo texto de entrada sempre gera o mesmo ID de chunk.
    """
    text = "Este é um texto de teste para verificar a estabilidade do hash."
    page1 = CleanedPageText(
        page_number=1,
        text=text,
        original_char_count=len(text),
        cleaned_char_count=len(text),
    )
    page2 = CleanedPageText(
        page_number=1,
        text=text,
        original_char_count=len(text),
        cleaned_char_count=len(text),
    )

    chunks1 = chunker.chunk_pages([page1])
    chunks2 = chunker.chunk_pages([page2])

    assert len(chunks1) == 1
    assert len(chunks2) == 1
    assert chunks1[0].chunk_id == chunks2[0].chunk_id
