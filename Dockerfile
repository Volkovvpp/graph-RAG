FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Pre-download the embedding model
ENV HF_HOME=/app/hf_cache
RUN uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
