from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Project
    PROJECT_NAME: str = "graph-rag"

    # ElasticSearch
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    ES_USER: str | None = None
    ES_PASSWORD: str | None = None
    ES_INDEX_kNOWLEDGE_BASE: str = "knowledge_base"

    # Neo4j
    NEO4J_HOST: str = "localhost"
    NEO4J_BOLT_PORT: int = 7687
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Hugging Face
    HF_API_TOKEN: str = ""
    HF_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    HF_INFERENCE_ENDPOINT: str | None = None

    @property
    def elastic_url(self) -> str:
        protocol = "https" if self.ES_USER and self.ES_PASSWORD else "http"
        return f"{protocol}://{self.ES_HOST}:{self.ES_PORT}"

    @property
    def neo4j_uri(self) -> str:
        return f"bolt://{self.NEO4J_HOST}:{self.NEO4J_BOLT_PORT}"

settings = Settings()

