import shutil
import tempfile
import asyncio
import json
import uuid
import os
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from src.core.config import settings
from src.retrieval.hybrid import HybridRetriever
from src.generation.synthesizer import Synthesizer

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REQUEST_TOPIC = settings.REQUEST_TOPIC
RESULT_TOPIC = settings.RESULT_TOPIC

producer: AIOKafkaProducer = None
results_queues: dict[str, asyncio.Queue] = {}


async def consume_results():
    consumer = AIOKafkaConsumer(
        RESULT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=f"api_group_{uuid.uuid4()}",
        auto_offset_reset="latest",
    )
    await consumer.start()
    try:
        async for msg in consumer:
            data = json.loads(msg.value.decode("utf-8"))
            task_id = data.get("task_id")
            if task_id in results_queues:
                await results_queues[task_id].put(data)
    finally:
        await consumer.stop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global producer
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    await producer.start()
    consumer_task = asyncio.create_task(consume_results())

    yield

    consumer_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer_task
    if producer:
        await producer.stop()



app = FastAPI(title="Graph RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

UPLOAD_TOPIC = getattr(settings, "UPLOAD_TOPIC", "upload_requests")

@app.post("/upload", status_code=202)
async def upload_document(file: UploadFile = File(...)):
    """Эндпоинт для асинхронной загрузки и обработки файла."""
    try:
        temp_dir = Path(tempfile.mkdtemp())
        file_path = temp_dir / file.filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        task_id = str(uuid.uuid4())
        msg = {
            "task_id": task_id,
            "type": "upload",
            "file_path": str(file_path),
            "filename": file.filename
        }
        await producer.send_and_wait(UPLOAD_TOPIC, json.dumps(msg).encode("utf-8"))

        return JSONResponse(
            status_code=202,
            content={"task_id": task_id, "message": "Файл принят в обработку"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_async", status_code=202)
async def generate_answer_async(request: QueryRequest):
    """Эндпоинт для генерации ответа через Kafka, возвращает 202."""
    task_id = str(uuid.uuid4())
    msg = {"task_id": task_id, "query": request.query}
    await producer.send_and_wait(REQUEST_TOPIC, json.dumps(msg).encode("utf-8"))
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "message": "Request accepted"},
    )


@app.websocket("/ws/results/{task_id}")
async def websocket_results(websocket: WebSocket, task_id: str):
    await websocket.accept()
    if task_id not in results_queues:
        results_queues[task_id] = asyncio.Queue()

    try:
        result = await results_queues[task_id].get()
        await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    finally:
        results_queues.pop(task_id, None)

@app.post("/query", response_model=QueryResponse)
async def generate_answer(request: QueryRequest):
    """Эндпоинт для генерации ответа на вопрос."""
    try:
        retriever = HybridRetriever()
        synthesizer = Synthesizer()

        retrieval_result = await retriever.retrieve(request.query)
        context = retrieval_result.get("context", "")

        answer = await synthesizer.generate_response(query=request.query, context=context)

        return QueryResponse(
            answer=answer,
            sources=retrieval_result.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))