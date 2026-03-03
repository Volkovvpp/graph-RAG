from sentence_transformers import SentenceTransformer
from src.core.config import settings
from src.core.logger import get_logger
import torch

logger = get_logger(__name__)

class EmbeddingClient:
    _instance: SentenceTransformer | None = None

    @classmethod
    def get_client(cls) -> SentenceTransformer:
        """Get the SentenceTransformer model instance."""
        if cls._instance is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading embedding model {settings.EMBEDDING_MODEL_ID} on device: {device}")
            try:
                cls._instance = SentenceTransformer(settings.EMBEDDING_MODEL_ID, device=device)
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise
        return cls._instance

    @classmethod
    def get_embedding(cls, text: str) -> list[float]:
        """Generate embedding for a single text."""
        model = cls.get_client()
        return model.encode(text, convert_to_tensor=False).tolist()

def get_embedding_client() -> SentenceTransformer:
    return EmbeddingClient.get_client()

