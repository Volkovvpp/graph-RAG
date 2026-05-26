from typing import List, Dict, Any
from src.connections.elastic import ElasticsearchClient
from src.core.config import settings
from src.core.logger import get_logger
from src.connections.embedder import EmbeddingClient

logger = get_logger(__name__)


class FullTextRetriever:
    """
    Handles full-text and vector search in Elasticsearch to find entry points for the graph traversal.
    """

    def __init__(self):
        self.index = settings.ES_INDEX_kNOWLEDGE_BASE
        self.embedding_client = EmbeddingClient()

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a standard full-text search (BM25) to find relevant chunks.

        Args:
            query: The user's search query.
            top_k: Number of results to return.

        Returns:
            List of dictionaries containing chunk data (id, text, score).
        """
        client = ElasticsearchClient.get_client()

        # Standard BM25 query
        body = {
            "size": top_k,
            "query": {"match": {"text": query}},
            "_source": ["chunk_id", "text", "metadata"],
        }

        try:
            response = await client.search(index=self.index, body=body)
            hits = response["hits"]["hits"]

            results = []
            for hit in hits:
                results.append(
                    {
                        "chunk_id": hit["_source"]["chunk_id"],
                        "text": hit["_source"]["text"],
                        "metadata": hit["_source"].get("metadata", {}),
                        "score": hit["_score"],
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []

    async def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a k-NN vector search to find the most similar chunks.

        Args:
            query: The user's search query.
            top_k: Number of results to return.

        Returns:
            List of dictionaries containing chunk data (id, text, score).
        """
        client = ElasticsearchClient.get_client()
        query_embedding = self.embedding_client.get_embedding(query)

        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": 100,  # Number of candidates to consider
        }

        try:
            response = await client.search(
                index=self.index,
                knn=knn_query,
                source_includes=["chunk_id", "text", "metadata"],
            )

            hits = response["hits"]["hits"]
            results = []
            for hit in hits:
                results.append(
                    {
                        "chunk_id": hit["_source"]["chunk_id"],
                        "text": hit["_source"]["text"],
                        "metadata": hit["_source"].get("metadata", {}),
                        "score": hit["_score"],
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Elasticsearch vector search failed: {e}")
            return []
