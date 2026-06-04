from typing import Any
from celery import Celery
from celery.signals import worker_process_init
from transformers import pipeline
import os

celery = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL",
                     "redis://redis:6379/0"),    # task queue
    backend=os.getenv("CELERY_RESULT_BACKEND",
                      "redis://redis:6379/1"),  # results
)

MODEL_NAME = os.getenv("WHISPER_MODEL")

_transcriber: Any | None = None


@worker_process_init.connect
def init_worker(**kwargs) -> None:
    global _transcriber

    try:
        _transcriber = pipeline(
            "automatic-speech-recognition", model=MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Could not load model '{MODEL_NAME}': {e}") from e


@celery.task(name="tasks.transcribe_audio", bind=True)
def transcribe_audio(self, audio_path: str):
    if _transcriber is None:
        raise RuntimeError("Transcriber is not initialized.")

    try:
        result_raw = _transcriber(audio_path)
    except Exception as e:
        raise RuntimeError(
            f"Transcription failed for '{audio_path}': {e}") from e

    text = result_raw["text"]

    try:
        celery.send_task(
            "rag_tasks.index_transcription",
            kwargs={
                "doc_id": self.request.id,
                "transcription": text,
                "source_path": audio_path,
            },
            queue="rag",
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not enqueue indexing task for '{audio_path}': {e}") from e

    return text
