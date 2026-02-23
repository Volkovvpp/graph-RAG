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
        Process a single chunk.
        """
        try:
            # 3. Extract Graph Data (Entities, Relationships)
            # Metadata might contain 'source' from loader
            source = chunk.metadata.get("source", "unknown")

            extraction_result = await self.extractor.extract(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata
            )

            chunk_id = extraction_result["chunk_id"]
            graph_data = extraction_result["graph_data"]

            # 4. Save to Elasticsearch (Text + Metadata)
            # We don't save the graph structure itself to ES, just the text and link
            es_task = ElasticsearchClient.index_document(
                chunk_id=chunk_id,
                text=chunk.page_content,
                metadata=chunk.metadata
            )

            # 5. Save to Neo4j (Graph Structure + Chunk Node)
            neo4j_task = Neo4jClient.save_chunk_graph(
                chunk_id=chunk_id,
                chunk_text=chunk.page_content,
                graph_data=graph_data,
                source=source
            )

            # Run saves concurrently
            await asyncio.gather(es_task, neo4j_task)

        except Exception as e:
            logger.error(f"Error processing chunk: {e}")

