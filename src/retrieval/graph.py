from typing import List, Dict, Any, Optional

from neo4j import Query

from src.connections.neo4j import Neo4jClient
from src.core.logger import get_logger

logger = get_logger(__name__)

class GraphRetriever:
    """
    Handles graph traversal in Neo4j to find connected context based on initial chunks.
    """

    @staticmethod
    async def get_connected_chunks(chunk_ids: List[str], hops: int = 1) -> List[Dict[str, Any]]:
        """
        Finds chunks that are semantically related to the input chunks by traversing the knowledge graph.

        Logic:
        1. Start from input Chunks.
        2. Traverse to interconnected Entities (MENTIONED relationship).
        3. From those Entities, traverse to OTHER Chunks that mention them.

        Args:
            chunk_ids: List of chunk UUIDs found by Elasticsearch.
            hops: Depth of traversal (currently 1 hop: Chunk -> Entity -> Chunk).

        Returns:
            List of unique relevant chunks (id, text) found in the graph.
        """
        if not chunk_ids:
            return []

        driver = Neo4jClient.get_driver()

        # Cypher query to find neighboring chunks via shared entities
        query = Query("""
        MATCH (start_chunk:Chunk)-[:MENTIONED]->(e:Entity)<-[:MENTIONED]-(connected_chunk:Chunk)
        WHERE start_chunk.id IN $chunk_ids
        AND start_chunk <> connected_chunk
        
        RETURN DISTINCT connected_chunk.id AS chunk_id, 
                        connected_chunk.text AS text,
                        e.name AS shared_entity,
                        e.type AS entity_type
        LIMIT 20
        """)

        try:
            results = []
            async with driver.session() as session:
                result = await session.run(query, chunk_ids=chunk_ids)

                async for record in result:
                    results.append({
                        "chunk_id": record["chunk_id"],
                        "text": record["text"],
                        "reasoning": f"Connected via {record['entity_type']}: {record['shared_entity']}"
                    })

            logger.info(f"Graph traversal found {len(results)} connected chunks for {len(chunk_ids)} input chunks.")
            return results

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []

    @staticmethod
    async def get_entity_context(chunk_ids: List[str]) -> List[str]:
        """
        Retrieves a summary of entities mentioned in the found chunks to provide extra context.
        Matches the entities directly connected to the chunks.
        """
        driver = Neo4jClient.get_driver()

        query = Query("""
        MATCH (c:Chunk)-[:MENTIONED]->(e:Entity)
        WHERE c.id IN $chunk_ids
        RETURN DISTINCT e.name AS name, e.description AS description, e.type AS type
        """)

        try:
            entities = []
            async with driver.session() as session:
                result = await session.run(query, chunk_ids=chunk_ids)
                async for record in result:
                    desc = record["description"]
                    if desc:
                        entities.append(f"{record['name']} ({record['type']}): {desc}")
                    else:
                        entities.append(f"{record['name']} ({record['type']})")

            return entities

        except Exception as e:
            logger.error(f"Entity context retrieval failed: {e}")
            return []

