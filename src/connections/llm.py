from huggingface_hub import AsyncInferenceClient
from src.core.config import settings


class HuggingFaceClient:
    @classmethod
    def get_client(cls) -> AsyncInferenceClient:
        # We recreate the client for each call or loop to avoid "Timeout context manager should be used inside a task"
        # and other event loop mismatch issues in Streamlit/Windows environments.
        if settings.HF_INFERENCE_ENDPOINT:
            return AsyncInferenceClient(
                model=settings.HF_INFERENCE_ENDPOINT, token=settings.HF_TOKEN
            )
        else:
            return AsyncInferenceClient(
                model=settings.HF_MODEL_ID, token=settings.HF_TOKEN
            )

    @classmethod
    async def verify_connectivity(cls) -> bool:
        """Verify connection to Hugging Face API"""
        async with cls.get_client() as client:
            try:
                # Simple chat completion to test the model (better for Instruct models)
                messages = [{"role": "user", "content": "Ping"}]
                await client.chat_completion(messages=messages, max_tokens=5)
                return True
            except Exception as e:
                print(f"Failed to connect to Hugging Face API: {e}")
                raise e

    @classmethod
    async def generate(cls, prompt: str, system_prompt: str = None) -> str:
        """Helper to generate text from the LLM."""
        async with cls.get_client() as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = await client.chat_completion(
                messages=messages, max_tokens=2048, temperature=0.9
            )
            return response.choices[0].message.content


async def get_llm_client() -> AsyncInferenceClient:
    return HuggingFaceClient.get_client()
