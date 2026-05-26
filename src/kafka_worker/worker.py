import asyncio
import json
import os
import logging
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from src.core.config import settings
from src.retrieval.hybrid import HybridRetriever
from src.generation.synthesizer import Synthesizer
from src.ingestion.pipeline import IngestionPipeline
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REQUEST_TOPIC = settings.REQUEST_TOPIC
RESULT_TOPIC = settings.RESULT_TOPIC
UPLOAD_TOPIC = getattr(settings, "UPLOAD_TOPIC", "upload_requests")


async def process_task(task_id: str, query: str, producer: AIOKafkaProducer):
    logger.info(f"Processing task {task_id}")
    try:
        retriever = HybridRetriever()
        synthesizer = Synthesizer()

        retrieval_result = await retriever.retrieve(query)
        context = retrieval_result.get("context", "")

        answer = await synthesizer.generate_response(query=query, context=context)

        result_data = {
            "task_id": task_id,
            "status": "completed",
            "answer": answer,
            "sources": retrieval_result.get("sources", []),
        }
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        result_data = {"task_id": task_id, "status": "error", "detail": str(e)}

    await producer.send_and_wait(RESULT_TOPIC, json.dumps(result_data).encode("utf-8"))
    logger.info(f"Task {task_id} completed and result sent to Kafka.")


async def process_upload_task(
    task_id: str, file_path_str: str, filename: str, producer: AIOKafkaProducer
):
    logger.info(f"Processing upload task {task_id} for file {filename}")
    try:
        pipeline = IngestionPipeline()
        file_path = Path(file_path_str)
        await pipeline.run(file_path)

        result_data = {
            "task_id": task_id,
            "status": "completed",
            "message": f"Файл '{filename}' успешно загружен и обработан.",
        }

        # Удаляем временный файл после обработки
        if file_path.exists():
            file_path.unlink()

    except Exception as e:
        logger.error(f"Error processing upload task {task_id}: {e}")
        result_data = {"task_id": task_id, "status": "error", "detail": str(e)}

    await producer.send_and_wait(RESULT_TOPIC, json.dumps(result_data).encode("utf-8"))
    logger.info(f"Upload task {task_id} completed and result sent to Kafka.")


async def consume():
    consumer = AIOKafkaConsumer(
        REQUEST_TOPIC,
        UPLOAD_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="query_workers",
        auto_offset_reset="earliest",
    )
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)

    await consumer.start()
    await producer.start()

    logger.info("Worker started, waiting for messages...")
    try:
        async for msg in consumer:
            data = json.loads(msg.value.decode("utf-8"))
            task_id = data.get("task_id")

            if not task_id:
                continue

            msg_type = data.get("type")
            if msg_type == "upload":
                file_path = data.get("file_path")
                filename = data.get("filename")
                if file_path and filename:
                    asyncio.create_task(
                        process_upload_task(task_id, file_path, filename, producer)
                    )
            else:
                query = data.get("query")
                if query:
                    asyncio.create_task(process_task(task_id, query, producer))
    finally:
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(consume())
