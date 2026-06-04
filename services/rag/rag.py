import os
import chromadb
import httpx
from celery import Celery
from celery.signals import worker_process_init
from sentence_transformers import SentenceTransformer

from schemas.models import SearchResultItem

celery = Celery(
    "rag_tasks",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
)

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))


_client = None
_collection = None
_embedder: SentenceTransformer | None = None


def init_chroma(host: str, port: int, collection_name: str) -> None:
    global _client, _collection, _embedder

    try:
        client = chromadb.HttpClient(host=host, port=port)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to ChromaDB at {host}:{port}: {e}") from e

    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        raise RuntimeError(
            f"Could not load embedding model '{EMBEDDING_MODEL}': {e}") from e

    _client = client
    _collection = collection
    _embedder = embedder


def add_transcription(doc_id: str, transcription: str, source_path: str) -> None:
    if _collection is None or _embedder is None:
        raise RuntimeError(
            "ChromaDB is not initialized. Call init_chroma() first.")

    try:
        embedding = _embedder.encode(transcription).tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding failed for doc '{doc_id}': {e}") from e

    try:
        _collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[transcription],
            metadatas=[{"source_path": source_path}],
        )
    except Exception as e:
        raise RuntimeError(
            f"ChromaDB upsert failed for doc '{doc_id}': {e}") from e


@worker_process_init.connect
def init_worker(**kwargs) -> None:
    init_chroma(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        collection_name=CHROMA_COLLECTION,
    )


@celery.task(name="rag_tasks.index_transcription")
def index_transcription(doc_id: str, transcription: str, source_path: str) -> dict[str, str]:
    add_transcription(
        doc_id=doc_id, transcription=transcription, source_path=source_path)
    return {"doc_id": doc_id, "status": "indexed"}


def query_transcriptions(query: str, top_k: int) -> list[SearchResultItem]:
    if _collection is None or _embedder is None:
        raise RuntimeError(
            "ChromaDB is not initialized. Call init_chroma() first.")

    try:
        query_embedding = _embedder.encode(query).tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding failed for query: {e}") from e

    try:
        raw = _collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise RuntimeError(f"ChromaDB query failed: {e}") from e

    # raw is batch-oriented: each value is a list-of-lists (one list per query)
    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    return [
        SearchResultItem(id=id_, document=doc, metadata=meta, distance=dist)
        for id_, doc, meta, dist in zip(ids, documents, metadatas, distances)
    ]


@celery.task(name="rag_tasks.search_transcriptions")
def search_transcriptions(query: str, top_k: int = 3) -> list[dict]:
    return [item.model_dump() for item in query_transcriptions(query=query, top_k=top_k)]


def _generate_answer(prompt: str) -> str:
    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt,
                  "stream": False, "options": {"temperature": 0}},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e


@celery.task(name="rag_tasks.answer_question")
def answer_question(question: str, top_k: int = 3) -> dict:
    sources = query_transcriptions(question, top_k)

    # Relevance guardrail: if no retrieved document clears the similarity
    # threshold, skip the LLM entirely to avoid hallucination.
    if not sources or min(s.distance for s in sources) > RELEVANCE_THRESHOLD:
        return {
            "answer": "Nie znalazłem w transkrypcjach informacji wystarczająco powiązanych z tym pytaniem.",
            "sources": [s.model_dump() for s in sources],
        }

    context = "\n".join(
        f"[{i+1}] {item.document}"
        for i, item in enumerate(sources)
    )

    prompt = (
        "Poniższe transkrypcje to wypowiedzi klientów banku zarejestrowane podczas rozmów.\n"
        "Na podstawie tych transkrypcji odpowiedz na pytanie — streszczając, "
        "co klienci mówili na dany temat.\n"
        "Odpowiadaj WYŁĄCZNIE na podstawie podanych transkrypcji. Nie korzystaj z wiedzy ogólnej.\n"
        "Jeśli żadna transkrypcja nie dotyczy pytanego tematu nawet pośrednio, "
        'odpowiedz dokładnie: "Nie znalazłem odpowiedzi w transkrypcjach."\n\n'
        f"Transkrypcje:\n{context}\n\n"
        f"Pytanie: {question}"
    )

    answer = _generate_answer(prompt)

    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
    }
