from typing import Dict, Any

from src.connections.llm import HuggingFaceClient
from src.generation.prompt_loader import get_prompt
from src.core.logger import get_logger

logger = get_logger(__name__)

class Synthesizer:
    """
    Synthesizes the final answer using the LLM and the retrieved context.
    """

    def __init__(self):
        self.system_prompt = "You are a helpful AI assistant. Answer based on the provided context."

    async def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response to the user's query using the provided context.

        Args:
            query: The user's question.
            context: The retrieved context string (text chunks + graph data).

        Returns:
            The generated answer string.
        """
        if not context:
            return "I couldn't find any relevant information to answer your question."

        try:
            # 1. Load and format the prompt
            prompt_template = get_prompt("rag_response")
            formatted_prompt = prompt_template.format(
                context=context,
                question=query
            )

            # 2. Call the LLM
            logger.info("Generating response with LLM...")
            response = await HuggingFaceClient.generate(
                prompt=formatted_prompt,
                system_prompt=self.system_prompt
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, an error occurred while generating the answer."

