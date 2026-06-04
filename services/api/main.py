from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from celery import Celery
import shutil
import os

from schemas.models import (
    HealthResponse,
    TranscribeResponse,
    JobStatusResponse,
    RagSearchRequest,
    RagSearchResponse,
    SearchResultItem,
    RagAnswerRequest,
    RagAnswerResponse
)

app = FastAPI()

ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/mpeg",
                         "audio/ogg", "audio/flac", "audio/x-wav"}

celery = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL",
                     "redis://redis:6379/0"),    # task queue
    backend=os.getenv("CELERY_RESULT_BACKEND",
                      "redis://redis:6379/1"),  # results
)


@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok", app="Audio RAG API", version="0.1.0")


@app.post("/audio/transcribe", status_code=202, response_model=TranscribeResponse)
def transcribe_audio(file: UploadFile = File(...)) -> TranscribeResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
        )

    tmp_path = f"/shared/{file.filename}"
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as e:
        raise HTTPException(
            status_code=500, detail=f"Could not save uploaded file: {e}")

    try:
        task = celery.send_task("tasks.transcribe_audio", args=[tmp_path])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not enqueue transcription task: {e}")

    return TranscribeResponse(task_id=task.id, status="processing")


@app.get("/audio/jobs/{task_id}", response_model=JobStatusResponse)
def get_audio_job(task_id: str) -> JobStatusResponse:
    task_result = celery.AsyncResult(task_id)
    state = task_result.state

    # Fast path: if the result backend has a real record, the task definitely exists.
    # This covers SUCCESS, FAILURE, STARTED, RETRY, REVOKED.
    if state != "PENDING":
        pass  # fall through to status mapping below
    else:
        # PENDING is ambiguous: either the task is waiting in the broker queue,
        # or the task_id was never submitted.
        # inspect().query_task() broadcasts a request to all workers asking
        # whether they currently hold this task (active or reserved).
        # timeout=1.0 means we wait at most 1 second for worker replies.
        worker_report = celery.control.inspect(timeout=1.0).query_task(task_id)
        # worker_report is None if no workers responded.
        # Otherwise: {"worker@host": {task_id: [state, info]}, ...}
        known_to_worker = worker_report is not None and any(
            task_id in tasks for tasks in worker_report.values()
        )
        if not known_to_worker:
            raise HTTPException(
                status_code=404, detail=f"Job '{task_id}' not found.")

    # https://docs.celeryq.dev/en/main/reference/celery.states.html

    if state in ("PENDING", "RECEIVED"):
        return JobStatusResponse(job_id=task_id, status="queued")

    if state in ("STARTED", "RETRY"):
        return JobStatusResponse(job_id=task_id, status="processing")

    if state == "SUCCESS":
        return JobStatusResponse(job_id=task_id, status="completed", transcription=task_result.result)

    # FAILURE, REVOKED, or any unknown custom state
    error = str(task_result.result) if isinstance(
        task_result.result, Exception) else state
    return JobStatusResponse(job_id=task_id, status="failed", error=error)


@app.post("/rag/search", response_model=RagSearchResponse)
def rag_search(req: RagSearchRequest) -> RagSearchResponse:
    try:
        task = celery.send_task(
            "rag_tasks.search_transcriptions",
            kwargs={"query": req.query, "top_k": req.top_k},
            queue="rag",
        )
        results = task.get(timeout=30)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return RagSearchResponse(
        query=req.query,
        results=[SearchResultItem(**item) for item in results],
    )


@app.post("/rag/answer", response_model=RagAnswerResponse)
def rag_answer(req: RagAnswerRequest) -> RagAnswerResponse:
    try:
        task = celery.send_task(
            "rag_tasks.answer_question",
            kwargs={"question": req.question, "top_k": req.top_k},
            queue="rag",
        )
        result = task.get(timeout=60)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Answer generation failed: {e}")

    return RagAnswerResponse(
        question=req.question,
        answer=result["answer"],
        sources=[SearchResultItem(**s) for s in result["sources"]],
    )
