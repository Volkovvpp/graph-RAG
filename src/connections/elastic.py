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
    async def close(cls):
        if cls._instance:
            await cls._instance.close()
            cls._instance = None

async def get_es_client() -> AsyncElasticsearch:
    return ElasticsearchClient.get_client()


