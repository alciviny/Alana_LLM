import pytest
from pathlib import Path
from src.alana_system.ingestion.audio_loader import AudioLoader, AudioDocument

# Usamos o tmp_path, um recurso do pytest para criar um diretório temporário
# para o teste, evitando sujar o sistema de arquivos.


def test_discover_audio_files(tmp_path: Path):
    """
    Verifica se o AudioLoader descobre corretamente os arquivos de áudio,
    incluindo subdiretórios e ignorando formatos não suportados.
    """
    # 1. Setup: Crie um ambiente de teste controlado
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Crie alguns arquivos de áudio válidos
    (raw_dir / "musica.mp3").write_text("dummy mp3 content")
    (raw_dir / "entrevista.WAV").write_text("dummy wav content")  # Teste de extensão maiúscula

    # Crie um subdiretório com um arquivo de áudio
    subdir = raw_dir / "podcasts"
    subdir.mkdir()
    (subdir / "episodio_1.m4a").write_text("dummy m4a content")

    # Crie arquivos que devem ser ignorados
    (raw_dir / "documento.txt").write_text("not an audio file")
    (raw_dir / "imagem.jpg").write_text("not an audio file")

    # 2. Execução: Chame o método que queremos testar
    loader = AudioLoader(raw_dir=str(raw_dir))
    discovered_docs = loader.discover()

    # 3. Asserção: Verifique se o resultado é o esperado
    assert len(discovered_docs) == 3, "Deveria encontrar 3 arquivos de áudio"

    # Verifique se os nomes dos arquivos foram capturados corretamente
    # Usamos um set para não depender da ordem em que os arquivos são encontrados
    discovered_names = {doc.name for doc in discovered_docs}
    expected_names = {"musica.mp3", "entrevista.WAV", "episodio_1.m4a"}
    assert discovered_names == expected_names


def test_audio_document_creation(tmp_path: Path):
    """
    Verifica se o AudioDocument é criado com os metadados corretos.
    """
    # 1. Setup
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    
    audio_content = "teste de conteudo para hash"
    audio_path = raw_dir / "teste.flac"
    audio_path.write_text(audio_content)

    # 2. Execução
    loader = AudioLoader(raw_dir=str(raw_dir))
    discovered_docs = loader.discover()

    # 3. Asserção
    assert len(discovered_docs) == 1
    doc = discovered_docs[0]

    assert isinstance(doc, AudioDocument)
    assert doc.name == "teste.flac"
    assert doc.format == ".flac"
    assert doc.size_bytes == len(audio_content.encode('utf-8'))
    assert doc.path.is_absolute()
    assert doc.path == audio_path.resolve()
    assert doc.duration_seconds is None  # O loader não deve preencher isso

    # Verifica se o ID (hash) é consistente
    import hashlib
    expected_hash = hashlib.sha256(audio_content.encode('utf-8')).hexdigest()
    assert doc.id == expected_hash


def test_loader_raises_error_for_nonexistent_dir():
    """
    Verifica se o AudioLoader levanta uma exceção se o diretório não existe,
    prevenindo erros inesperados mais tarde.
    """
    # O pytest.raises funciona como um "try/except" para testes.
    # O teste passa se o código dentro do 'with' levantar a exceção esperada.
    with pytest.raises(FileNotFoundError):
        AudioLoader(raw_dir="caminho/que/nao/existe")
