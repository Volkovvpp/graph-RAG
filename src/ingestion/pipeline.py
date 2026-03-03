import asyncio
from pathlib import Path
from typing import List, Union

from src.core.logger import get_logger
from src.ingestion.loader import DocumentLoader
from src.ingestion.splitter import TextSplitter
from src.ingestion.extractor import GraphExtractor
from src.connections.elastic import ElasticsearchClient
from src.connections.neo4j import Neo4jClient

logger = get_logger(__name__)

class IngestionPipeline:
    """
    Orchestrates the ingestion process:
    Load -> Split -> Extract -> Save (Elastic + Neo4j)
    """

    def __init__(self):
        self.extractor = GraphExtractor()

    async def run(self, source_path: Union[str, Path]):
        """
        Run the full ingestion pipeline on a file or directory.
        """
        logger.info(f"Starting ingestion for: {source_path}")

        # 1. Load Documents
        try:
            if Path(source_path).is_dir():
                docs = DocumentLoader.load_directory(source_path)
            else:
                docs = DocumentLoader.load_file(source_path)
            logger.info(f"Loaded {len(docs)} documents.")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return

        # 2. Split into Chunks
        chunks = TextSplitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks.")

        # Ensure indices exist
        await ElasticsearchClient.create_index()

        # 3. Process Chunks (Extract & Save)
        # We process chunks in parallel batches to avoid overwhelming LLM or DBs
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            await self._process_batch(batch)
            logger.info(f"Processed batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

        logger.info("Ingestion complete.")

    async def _process_batch(self, chunks):
        """
        Process a batch of chunks: Extract -> Save to ES & Neo4j
        """
        tasks = [self._process_single_chunk(chunk) for chunk in chunks]
        await asyncio.gather(*tasks)

    async def _process_single_chunk(self, chunk):
        """
        Process a single chunk with transactional behavior.
        1. Extract graph data.
        2. Save to Elasticsearch.
        3. Save to Neo4j.
        4. If Neo4j save fails, delete from Elasticsearch to roll back.
        """
        chunk_id = None
        try:
            # 3. Extract Graph Data
            source = chunk.metadata.get("source", "unknown")
            extraction_result = await self.extractor.extract(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata
            )
            chunk_id = extraction_result["chunk_id"]
            graph_data = extraction_result["graph_data"]

            # 4. Save to Elasticsearch first
            await ElasticsearchClient.index_document(
                chunk_id=chunk_id,
                text=chunk.page_content,
                metadata=chunk.metadata
            )
            logger.debug(f"Indexed chunk {chunk_id} in Elasticsearch.")

            # 5. Save to Neo4j
            await Neo4jClient.save_chunk_graph(
                chunk_id=chunk_id,
                chunk_text=chunk.page_content,
                graph_data=graph_data,
                source=source
            )
            logger.debug(f"Saved graph for chunk {chunk_id} in Neo4j.")

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id if chunk_id else 'unknown'}: {e}. Rolling back...")
            # If the error happened after saving to ES, roll back
            if chunk_id:
                logger.info(f"Rolling back Elasticsearch document for chunk {chunk_id}.")
                await ElasticsearchClient.delete_document(chunk_id)
            # Re-raise the exception to notify the caller (e.g., gather)
            raise
