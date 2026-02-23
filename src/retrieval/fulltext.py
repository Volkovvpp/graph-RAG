from typing import List, Dict, Any
from src.connections.elastic import ElasticsearchClient
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

class FullTextRetriever:
    """
    Handles full-text search in Elasticsearch to find entry points for the graph traversal.
    """

    def __init__(self):
        self.index = settings.ES_INDEX_kNOWLEDGE_BASE

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
            "query": {
                "match": {
                    "text": query
                }
            },
            "_source": ["chunk_id", "text", "metadata"]
        }

        try:
            response = await client.search(index=self.index, body=body)
            hits = response["hits"]["hits"]

            results = []
            for hit in hits:
                results.append({
                    "chunk_id": hit["_source"]["chunk_id"],
                    "text": hit["_source"]["text"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"]
                })

            return results

        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []

