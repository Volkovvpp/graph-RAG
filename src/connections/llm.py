from huggingface_hub import AsyncInferenceClient
from src.core.config import settings

class HuggingFaceClient:
    _client: AsyncInferenceClient | None = None

    @classmethod
    def get_client(cls) -> AsyncInferenceClient:
        if cls._client is None:
            # If a dedicated endpoint is provided, use it as the base URL
            # Otherwise use the model ID for the public Serverless Inference API
            if settings.HF_INFERENCE_ENDPOINT:
                 # Initialize with valid endpoint URL
                 # Note: model argument can be None if the endpoint is model-specific
                 cls._client = AsyncInferenceClient(model=settings.HF_INFERENCE_ENDPOINT, token=settings.HF_API_TOKEN)
            else:
                 cls._client = AsyncInferenceClient(model=settings.HF_MODEL_ID, token=settings.HF_API_TOKEN)
        return cls._client

    @classmethod
    async def verify_connectivity(cls) -> bool:
        """Verify connection to Hugging Face API"""
        client = cls.get_client()
        try:
            # Simple chat completion to test the model (better for Instruct models)
            messages = [{"role": "user", "content": "Ping"}]
            await client.chat_completion(
                messages=messages,
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"Failed to connect to Hugging Face API: {e}")
            raise e

async def get_llm_client() -> AsyncInferenceClient:
    return HuggingFaceClient.get_client()


