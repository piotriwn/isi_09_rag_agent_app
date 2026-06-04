from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class TranscribeResponse(BaseModel):
    task_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    transcription: str | None = None
    error: str | None = None


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=[
                       "jak sprawdzić stan konta"])
    top_k: int = Field(3, ge=1, le=10, examples=[5])


class SearchResultItem(BaseModel):
    id: str
    document: str
    metadata: dict
    distance: float


class RagSearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class RagAnswerRequest(BaseModel):
    question: str = Field(..., min_length=5, examples=[
                          "jak zablokować kartę?"])
    top_k: int = Field(3, ge=1, le=10, examples=[5])


class RagAnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchResultItem]
