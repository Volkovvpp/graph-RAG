from neo4j import AsyncGraphDatabase, AsyncDriver, Query
from src.core.config import settings


class Neo4jClient:
    _driver: AsyncDriver | None = None

    @classmethod
    def get_driver(cls) -> AsyncDriver:
        if cls._driver is None:
            uri = settings.neo4j_uri
            auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)

            # Create driver
            cls._driver = AsyncGraphDatabase.driver(uri, auth=auth)

        return cls._driver

    @classmethod
    async def save_chunk_graph(
        cls, chunk_id: str, chunk_text: str, graph_data: dict, source: str = "unknown"
    ):
        """
        Save a chunk and its extracted graph data into Neo4j.
        Links Chunk -> Entity and Entity -> Entity.
        """
        driver = cls.get_driver()

        query = Query("""
        // 1. Create Chunk node
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text, c.source = $source
        
        // 2. Process Entities
        FOREACH (entity IN $entities |
            MERGE (e:Entity {name: entity.name})
            SET e.type = entity.type, e.description = entity.description
            
            // Link Chunk to Entity
            MERGE (c)-[:MENTIONED]->(e)
        )
        
        // 3. Process Relationships between Entities
        FOREACH (rel IN $relationships |
            MERGE (s:Entity {name: rel.source})
            MERGE (t:Entity {name: rel.target})
            MERGE (s)-[:RELATED {type: rel.relation_type}]->(t)
        )
        """)

        try:
            async with driver.session() as session:
                await session.run(
                    query,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    entities=graph_data.get("entities", []),
                    relationships=graph_data.get("relationships", []),
                )
        except Exception as e:
            print(f"Failed to save graph data for chunk {chunk_id}: {e}")
            raise e

    @classmethod
    async def close(cls):
        if cls._driver:
            await cls._driver.close()
            cls._driver = None

    @classmethod
    async def verify_connectivity(cls):
        """Verify database connectivity"""
        driver = cls.get_driver()
        try:
            await driver.verify_connectivity()

            # Additional check: execute a simple query
            # If neo4j version is old or there are permission issues
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS num")
                record = await result.single()
                if record["num"] != 1:
                    raise ConnectionError("Neo4j test query returned unexpected result")
            return True
        except Exception as e:
            # Log or re-raise error
            print(f"Failed to connect to Neo4j: {e}")
            raise e


async def get_neo4j_driver() -> AsyncDriver:
    return Neo4jClient.get_driver()
