"""Deterministic loading, token splitting and corpus snapshots."""

import hashlib
import json
import os
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_documents(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"No existe el directorio: {directory}")
    result = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".pdf"}:
            continue
        source = path.relative_to(directory).as_posix()
        if path.suffix.lower() == ".pdf":
            pages = [(p.extract_text() or "", i + 1) for i, p in enumerate(PdfReader(path).pages)]
            records = [{"text": text, "page": page} for text, page in pages]
        elif path.suffix.lower() == ".md":
            records = [{"text": path.read_text(encoding="utf-8"), "page": 1}]
        else:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"{source}: JSON debe ser una lista de registros")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{source}: registro inválido")
            text = record.get("text", "")
            tags = record.get("categories", [path.stem])
            page = record.get("page", 1)
            doc_id = record.get("document_id", source)
            if (
                not isinstance(text, str)
                or not text.strip()
                or not isinstance(tags, list)
                or not tags
                or not all(isinstance(tag, str) and tag.strip() for tag in tags)
                or not isinstance(doc_id, str)
                or not doc_id.strip()
                or isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
            ):
                raise ValueError(f"{source}: texto, document_id, página o categorías inválidos")
            result.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "document_id": doc_id,
                        "source": source,
                        "page": page,
                        "categories": tags,
                    },
                )
            )
    if not result:
        raise ValueError(
            "No hay documentos compatibles con texto extraíble (PDF escaneado requiere OCR)"
        )
    # One logical document may span pages, but IDs may not alias different files.
    owners = {}
    for doc in result:
        key = doc.metadata["document_id"]
        if key in owners and owners[key] != doc.metadata["source"]:
            raise ValueError(f"document_id duplicado entre archivos: {key}")
        owners[key] = doc.metadata["source"]
    return result


def build_chunks(settings):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    encoder = tiktoken.get_encoding("cl100k_base")

    def bound(chunk):
        # Token counts are not additive at separator boundaries. Verify actual content.
        if len(encoder.encode(chunk.page_content, disallowed_special=())) <= settings.chunk_size:
            return [chunk]
        midpoint = len(chunk.page_content) // 2
        if midpoint == 0:
            raise ValueError("CHUNK_SIZE demasiado pequeño para un carácter Unicode")
        left = Document(page_content=chunk.page_content[:midpoint], metadata=dict(chunk.metadata))
        right = Document(page_content=chunk.page_content[midpoint:], metadata=dict(chunk.metadata))
        return bound(left) + bound(right)

    chunks = [
        part
        for chunk in splitter.split_documents(load_documents(settings.data_dir))
        for part in bound(chunk)
    ]
    for i, chunk in enumerate(chunks):
        identity = json.dumps([chunk.metadata, i, chunk.page_content], sort_keys=True)
        chunk.metadata["chunk_id"] = digest(identity)
    return chunks


def make_manifest(settings, chunks):
    rows = [{"text": d.page_content, "metadata": d.metadata} for d in chunks]
    corpus_hash = digest(json.dumps([settings.model, settings.dimension, rows], sort_keys=True))
    return {
        "version": 1,
        "index_name": settings.index_name,
        "model": settings.model,
        "dimension": settings.dimension,
        "namespace_prefix": settings.namespace,
        "namespace": f"{settings.namespace}-{corpus_hash[:24]}",
        "corpus_hash": corpus_hash,
        "chunks": rows,
    }


def save_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_manifest(settings):
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["index_name"] != settings.index_name
        or manifest["model"] != settings.model
        or manifest["dimension"] != settings.dimension
        or manifest["namespace_prefix"] != settings.namespace
    ):
        raise ValueError("Configuración incompatible con el corpus; ejecutar ingest.py")
    docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in manifest["chunks"]]
    if make_manifest(settings, docs) != manifest:
        raise ValueError("Manifest alterado o incompatible; ejecutar ingest.py")
    return manifest, docs
