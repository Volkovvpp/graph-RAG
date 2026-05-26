import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.generation.synthesizer import Synthesizer

app = FastAPI(title="Graph RAG API")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.post("/upload", status_code=202)
async def upload_document(file: UploadFile = File(...)):
    """Эндпоинт для асинхронной загрузки и обработки файла."""
    try:
        temp_dir = Path(tempfile.mkdtemp())

        # fix: assure file.filename is valid string
        filename = file.filename or "unknown_file"
        file_path = temp_dir / filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = IngestionPipeline()
        await pipeline.run(file_path)

        return {"message": f"Файл '{file.filename}' успешно загружен и обработан."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def generate_answer(request: QueryRequest):
    """
    Эндпоинт для генерации ответа на вопрос.
    """
    try:
        retriever = HybridRetriever()
        synthesizer = Synthesizer()

        # Поиск релевантного контекста
        retrieval_result = await retriever.retrieve(request.query)
        context = retrieval_result.get("context", "")

        # Генерация ответа
        answer = await synthesizer.generate_response(
            query=request.query, context=context
        )

        return QueryResponse(answer=answer, sources=retrieval_result.get("sources", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
