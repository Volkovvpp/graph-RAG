from neo4j import AsyncGraphDatabase, AsyncDriver
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

