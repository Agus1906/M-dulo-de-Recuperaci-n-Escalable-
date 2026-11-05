"""Pinecone Serverless lifecycle, bounded readiness and idempotent batches."""

import logging
import time

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def transient(exc):
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    return isinstance(exc, (TimeoutError, ConnectionError)) or status in {429, 500, 502, 503, 504}


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception(transient),
    reraise=True,
)
def upsert_batch(index, vectors, namespace):
    index.upsert(vectors=vectors, namespace=namespace)


def embeddings(settings):
    return OpenAIEmbeddings(
        model=settings.model, dimensions=settings.dimension, max_retries=3, request_timeout=60
    )


def ensure_index(settings, client=None, timeout=120):
    client = client or Pinecone()
    if settings.index_name not in client.list_indexes().names():
        client.create_index(
            name=settings.index_name,
            dimension=settings.dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )
    deadline = time.monotonic() + timeout
    while True:
        info = client.describe_index(settings.index_name)
        if info.dimension != settings.dimension or info.metric != "cosine":
            raise ValueError("Índice incompatible: se requiere dimensión 1536 y métrica cosine")
        spec = info.spec.to_dict() if hasattr(info.spec, "to_dict") else info.spec
        if not spec.get("serverless"):
            raise ValueError("El índice existente no es Serverless")
        if info.status["ready"]:
            return client.Index(host=info.host)
        if time.monotonic() >= deadline:
            raise TimeoutError("El índice no estuvo listo dentro del plazo")
        time.sleep(2)


def wait_visible(index, ids, namespace, timeout=120):
    deadline = time.monotonic() + timeout
    pending = set(ids)
    while pending:
        batch = sorted(pending)[:100]
        response = index.fetch(ids=batch, namespace=namespace)
        pending.difference_update(response.vectors.keys())
        if pending and time.monotonic() >= deadline:
            raise TimeoutError("Pinecone aún no expone todos los chunks; reejecutar ingesta")
        if pending:
            time.sleep(1)


def ingest(settings):
    from .documents import build_chunks, make_manifest, save_manifest

    chunks = build_chunks(settings)
    manifest = make_manifest(settings, chunks)
    index = ensure_index(settings)
    embedder = embeddings(settings)
    for start in range(0, len(chunks), 64):
        batch = chunks[start : start + 64]
        values = embedder.embed_documents([d.page_content for d in batch])
        if len(values) != len(batch) or any(len(v) != settings.dimension for v in values):
            raise ValueError("El proveedor devolvió embeddings con dimensiones/cantidad inválidas")
        vectors = [
            {
                "id": doc.metadata["chunk_id"],
                "values": vector,
                "metadata": {**doc.metadata, "text": doc.page_content},
            }
            for doc, vector in zip(batch, values)
        ]
        upsert_batch(index, vectors, manifest["namespace"])
        logger.info("Ingestados %d/%d chunks", min(start + 64, len(chunks)), len(chunks))
    wait_visible(index, [d.metadata["chunk_id"] for d in chunks], manifest["namespace"])
    save_manifest(settings.manifest_path, manifest)
    logger.info("Corpus publicado: %d chunks, namespace=%s", len(chunks), manifest["namespace"])
    return manifest
