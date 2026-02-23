from typing import List, Dict, Any, Set
import asyncio
from src.retrieval.fulltext import FullTextRetriever
from src.retrieval.graph import GraphRetriever
from src.core.logger import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    """
    Combines Full-Text Search (Elasticsearch) and Graph Traversal (Neo4j)
    to provide a rich context window for RAG.
    """

    def __init__(self):
        self.fulltext = FullTextRetriever()
        self.graph = GraphRetriever()

    async def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute hybrid retrieval:
        1. Find relevant chunks via Elastic (text similarity).
        2. Expand context by finding connected chunks in Neo4j.
        3. Gather entity descriptions for better understanding.

        Args:
            query: User question.
            top_k: Number of initial Full-Text results.

        Returns:
            Dictionary with 'context' string and source references.
        """
        logger.info(f"Starting hybrid retrieval for query: {query}")

        # 1. Full-Text Search (Seed Chunks)
        es_results = await self.fulltext.search(query, top_k=top_k)
        if not es_results:
            logger.warning("No results found in Elasticsearch.")
            return {"context": "", "sources": []}

        seed_chunk_ids = [res["chunk_id"] for res in es_results]

        # 2. Graph Expansion (Parallel)
        # We fetch connected chunks and entity info concurrently
        graph_chunks_task = self.graph.get_connected_chunks(seed_chunk_ids, hops=1)
        entity_context_task = self.graph.get_entity_context(seed_chunk_ids)

        graph_chunks, entity_context = await asyncio.gather(graph_chunks_task, entity_context_task)

        # 3. Combine & Deduplicate Results
        # Use a dictionary to handle duplicates by chunk_id
        final_chunks: Dict[str, str] = {}

        # Add ES chunks (High priority)
        for res in es_results:
            final_chunks[res["chunk_id"]] = res["text"]

        # Add Graph chunks (Contextual expansion)
        for g_res in graph_chunks:
            # Only add if not already present (or logic could specificy to append reasoning)
            if g_res["chunk_id"] not in final_chunks:
                final_chunks[g_res["chunk_id"]] = g_res["text"]

        logger.info(f"Retrieved {len(final_chunks)} unique chunks ({len(es_results)} from ES, {len(graph_chunks)} from Graph).")

        # 4. Construct Context String for LLM
        context_parts = []

        # Add Entity Definitions first (Knowledge Graph definitions)
        if entity_context:
            context_parts.append("--- KNOWLEDGE GRAPH ENTITIES ---")
            context_parts.extend(entity_context)
            context_parts.append("") # Blank line

        # Add Document Chunks
        context_parts.append("--- RELEVANT DOCUMENT EXCERPTS ---")
        for i, (cid, text) in enumerate(final_chunks.items(), 1):
            context_parts.append(f"Excerpt {i}: {text}\n")

        full_context = "\n".join(context_parts)

        return {
            "context": full_context,
            "chunk_ids": list(final_chunks.keys()),
            "sources": es_results + graph_chunks # Raw data for citations if needed
        }

