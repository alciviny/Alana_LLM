from qdrant_client import QdrantClient
client = QdrantClient(location=':memory:')
print(f'Has search: {hasattr(client, "search")}')
print(f'Type of client: {type(client)}')
