from unittest.mock import MagicMock, patch
from qdrant_client.models import ScoredPoint

from alana_system.embeddings.embedder import TextEmbedder
from alana_system.memory.vector_store import VectorStore


def test_vector_store():
    """
    Testa a lógica do VectorStore com um cliente Qdrant mockado para evitar
    a necessidade de uma instância real do Qdrant em execução.
    """
    # 1. Mock do QdrantClient
    #    Patcheamos a classe no módulo ONDE ELA É USADA.
    mock_client_instance = MagicMock()
    with patch(
        "alana_system.memory.vector_store.QdrantClient",
        return_value=mock_client_instance,
    ):
        # 2. Configuração do comportamento do mock
        # Acessado no __init__ do VectorStore para verificar se a collection existe.
        # Retornamos uma resposta vazia para simular que não existe e forçar a criação.
        mock_client_instance.get_collections.return_value = MagicMock(collections=[])

        # Acessado pelo método `search` do VectorStore.
        # Simulamos o retorno de um ponto com score alto.
        mock_search_response = [
            ScoredPoint(
                id="b" * 32,
                version=1,
                score=0.95,
                payload={
                    "original_id": "b" * 64,
                    "page_number": 1,
                    "text": "Alavancagem excessiva pode causar perdas",
                },
            )
        ]
        mock_client_instance.search.return_value = mock_search_response

        # 3. Execução da lógica do teste
        # O TextEmbedder ainda funciona normalmente, baixando o modelo.
        # Isso está ok, pois queremos testar a integração com embeddings reais.
        embedder = TextEmbedder()
        store = VectorStore(collection_name="test_docs", vector_dim=384)

        # O teste original usava um FakeChunk, vamos usar o mesmo princípio
        # para criar os dados de entrada.
        class FakeChunk:
            def __init__(self, cid, text):
                self.chunk_id = cid
                self.page_number = 1
                self.text = text

        fake_chunks = [
            FakeChunk("a" * 64, "Risco financeiro em mercados voláteis"),
            FakeChunk("b" * 64, "Alavancagem excessiva pode causar perdas"),
        ]

        # O embedder transforma os chunks em `EmbeddedChunk`, que têm o vetor.
        embedded = embedder.embed_chunks(fake_chunks)
        # A chamada a `upsert_embeddings` agora usará o mock do cliente.
        store.upsert_embeddings(embedded)

        query = "Quais os riscos da alavancagem?"
        q_emb = embedder.model.encode(query, normalize_embeddings=True)

        # A chamada a `search` também usará o mock.
        results = store.search(q_emb)

        # 4. Asserts
        # Verificamos se o resultado do mock foi processado e retornado corretamente.
        assert len(results) > 0
        assert results[0]["text"] == "Alavancagem excessiva pode causar perdas"
        assert results[0]["score"] > 0.9

        # Verificamos se os métodos do cliente mockado foram chamados como esperado.
        mock_client_instance.get_collections.assert_called_once()
        mock_client_instance.create_collection.assert_called_once()
        mock_client_instance.upsert.assert_called_once()
        mock_client_instance.search.assert_called_once()
