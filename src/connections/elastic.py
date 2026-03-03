from elasticsearch import AsyncElasticsearch
from src.core.config import settings

class ElasticsearchClient:
    _instance: AsyncElasticsearch | None = None

    @classmethod
    def get_client(cls) -> AsyncElasticsearch:
        if cls._instance is None:
            es_url = settings.elastic_url

            basic_auth = None
            if settings.ES_USER and settings.ES_PASSWORD:
                basic_auth = (settings.ES_USER, settings.ES_PASSWORD)

            cls._instance = AsyncElasticsearch(
                hosts=es_url,
                basic_auth=basic_auth,
                verify_certs=False,
            )
        return cls._instance

    @classmethod
    async def create_index(cls):
        """Create the knowledge base index with appropriate mapping if it doesn't exist."""
        client = cls.get_client()
        index_name = settings.ES_INDEX_kNOWLEDGE_BASE

        if not await client.indices.exists(index=index_name):
            # Define mapping for vector search + text search
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "chunk_id": {"type": "keyword"},
                        "metadata": {"type": "object"},
                        "vector": {
                            "type": "dense_vector",
                            "dims": 384, # Assuming a small embedding model later, for now just placeholder
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
            # For now we skip vector mapping as we haven't set up embeddings
            # and just use standard text search
            simple_mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "chunk_id": {"type": "keyword"},
                        "metadata": {"type": "object"}
                    }
                }
            }
            await client.indices.create(index=index_name, body=simple_mapping)
            print(f"Created index: {index_name}")

    @classmethod
    async def index_document(cls, chunk_id: str, text: str, metadata: dict):
        """Index a single document/chunk."""
        client = cls.get_client()
        index_name = settings.ES_INDEX_kNOWLEDGE_BASE

        document = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            # "vector": ... (add embedding later)
        }

        await client.index(index=index_name, id=chunk_id, document=document)

    @classmethod
    async def delete_document(cls, chunk_id: str):
        """Delete a single document/chunk by its ID."""
        client = cls.get_client()
        index_name = settings.ES_INDEX_kNOWLEDGE_BASE

        try:
            await client.delete(index=index_name, id=chunk_id)
        except Exception as e:
            # Log error if deletion fails, but don't re-raise
            # as the main error is the one that triggered the rollback.
            print(f"Failed to delete document {chunk_id} during rollback: {e}")

    @classmethod
    async def close(cls):
        if cls._instance:
            await cls._instance.close()
            cls._instance = None

async def get_es_client() -> AsyncElasticsearch:
    return ElasticsearchClient.get_client()
