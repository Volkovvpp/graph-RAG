import json
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

from src.connections.llm import HuggingFaceClient
from src.core.logger import get_logger
from src.generation.prompt_loader import get_prompt

logger = get_logger(__name__)

class GraphEntity(BaseModel):
    name: str
    type: str
    description: str = ""

class GraphRelationship(BaseModel):
    source: str
    target: str
    relation_type: str
    description: str = ""

class ExtractedGraphData(BaseModel):
    entities: List[GraphEntity]
    relationships: List[GraphRelationship]

class GraphExtractor:
    """
    Extracts entities and relationships from text using an LLM
    to build a graph structure.
    """

    def __init__(self):
        pass

    def _create_extraction_prompt(self, text: str) -> str:
        return get_prompt("extraction").format(text=text)

    async def extract(self, chunk_text: str, chunk_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Processes a single chunk of text and returns specific graph data
        plus the chunk ID to link ES and Neo4j.
        """
        # 1. Provide a stable ID for both ES and Neo4j
        chunk_id = str(uuid.uuid4())

        prompt = self._create_extraction_prompt(chunk_text)

        try:
            # We use the class method directly as we added it to HuggingFaceClient
            response_text = await HuggingFaceClient.generate(prompt)

            # fast cleanup
            clean_json = response_text.replace("```json", "").replace("```", "").strip()

            # Sometimes models add preamble text, try to find the first { and last }
            start_idx = clean_json.find("{")
            end_idx = clean_json.rfind("}")
            if start_idx != -1 and end_idx != -1:
                clean_json = clean_json[start_idx:end_idx+1]

            data = json.loads(clean_json)

            validated_data = ExtractedGraphData(**data)

            return {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata or {},
                "graph_data": validated_data.model_dump()
            }

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to parse graph data from chunk (JSON/Schema error): {e}. Saving chunk without graph data.")
            return {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata or {},
                "graph_data": {"entities": [], "relationships": []}
            }
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}")
             # Return valid structure even on failure so pipeline doesn't break
            return {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata or {},
                "graph_data": {"entities": [], "relationships": []}
            }
