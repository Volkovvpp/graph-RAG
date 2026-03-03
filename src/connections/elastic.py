from elasticsearch import AsyncElasticsearch
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

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
    async def create_index(cls, client: AsyncElasticsearch):
        """Create the knowledge base index with appropriate mapping if it doesn't exist."""
        index_name = settings.ES_INDEX_kNOWLEDGE_BASE

        if not await client.indices.exists(index=index_name):
            mapping = {
                "properties": {
                    "text": {"type": "text"},
                    "chunk_id": {"type": "keyword"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": settings.EMBEDDING_DIM
                    }
                }
            }
            try:
                await client.indices.create(index=index_name, mappings=mapping)
                logger.info(f"Created index: {index_name} with embedding dimension {settings.EMBEDDING_DIM}")
            except Exception as e:
                if "resource_already_exists_exception" not in str(e):
                    raise

    @classmethod
    async def index_document(cls, client: AsyncElasticsearch, chunk_id: str, text: str, metadata: dict, embedding: list[float]):
        """Index a single document/chunk with its embedding."""
        index_name = settings.ES_INDEX_kNOWLEDGE_BASE

        document = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            "embedding": embedding,
        }

        await client.index(index=index_name, id=chunk_id, document=document)

    @classmethod
    async def delete_document(cls, client: AsyncElasticsearch, chunk_id: str):
        """Delete a single document/chunk by its ID."""
        index_name = settings.ES_INDEX_kNOWLEDGE_BASE

        try:
            await client.delete(index=index_name, id=chunk_id)
        except Exception as e:
            logger.error(f"Failed to delete document {chunk_id} during rollback: {e}")

    @classmethod
    async def close(cls):
        if cls._instance:
            await cls._instance.close()
            cls._instance = None

async def get_es_client() -> AsyncElasticsearch:
    return ElasticsearchClient.get_client()
