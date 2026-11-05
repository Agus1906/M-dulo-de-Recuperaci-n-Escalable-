import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.retrievers import BaseRetriever

from rag_system.cloud import ensure_index, ingest, wait_visible
from rag_system.config import Settings
from rag_system.documents import (
    build_chunks,
    load_documents,
    make_manifest,
    read_manifest,
    save_manifest,
)
from rag_system.evaluation import evaluate, load_golden, metrics
from rag_system.retrieval import PineconeRetriever, RAGSystem


@pytest.fixture
def settings(tmp_path):
    return Settings(index_name="test-index", manifest_path=tmp_path / "corpus.json")


@pytest.fixture
def corpus(settings):
    chunks = build_chunks(settings)
    save_manifest(settings.manifest_path, make_manifest(settings, chunks))
    return chunks


class FakeSemantic(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query, *, run_manager):
        return self.docs


def test_deterministic_chunks_and_manifest(settings):
    chunks = build_chunks(settings)
    assert len(chunks) >= 8
    assert make_manifest(settings, chunks) == make_manifest(settings, build_chunks(settings))
    assert all(
        {"document_id", "source", "page", "categories", "chunk_id"} <= d.metadata.keys()
        for d in chunks
    )
    changed = chunks[0].model_copy(deep=True)
    changed.page_content += " cambio"
    assert (
        make_manifest(settings, [changed])["namespace"]
        != make_manifest(settings, chunks)["namespace"]
    )


def test_manifest_roundtrip_and_config_mismatch(settings, corpus):
    assert read_manifest(settings)[1] == corpus
    with pytest.raises(ValueError):
        read_manifest(replace(settings, namespace="other"))
    data = json.loads(settings.manifest_path.read_text())
    data["chunks"][0]["text"] = "tampered"
    settings.manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_manifest(settings)


@pytest.mark.parametrize(
    "kwargs", [{"dimension": 512}, {"namespace": ""}, {"chunk_overlap": 650}, {"chunk_size": 900}]
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Settings(index_name="test", **kwargs)


def test_invalid_json_and_empty_input(tmp_path):
    with pytest.raises(ValueError):
        load_documents(tmp_path)
    (tmp_path / "bad.json").write_text('[{"text": "x", "categories": "wrong"}]')
    with pytest.raises(ValueError):
        load_documents(tmp_path)


def test_markdown(tmp_path):
    (tmp_path / "example.md").write_text("# Ejemplo\ncontenido", encoding="utf-8")
    doc = load_documents(tmp_path)[0]
    assert doc.metadata["page"] == 1
    assert doc.metadata["document_id"] == "example.md"


def test_metric_definitions():
    assert metrics(["a", "a", "b"], ["a"], 5) == {"precision": 0.2, "recall": 1.0, "hits": 1}
    assert metrics(["x"], ["a"], 5)["recall"] == 0
    assert metrics(["a", "b"], ["a", "b", "c"], 5)["recall"] == pytest.approx(2 / 3)
    assert metrics(["x", "y", "a"], ["a"], 2)["precision"] == 0


def test_hybrid_unique_sources_and_evaluation(settings, corpus):
    system = RAGSystem(settings, semantic=FakeSemantic(docs=corpus))
    docs = system.retrieve("asyncio Semaphore")
    assert len(docs) == 5
    assert len({d.metadata["document_id"] for d in docs}) == 5
    assert system.retriever.weights == [0.5, 0.5]
    with pytest.raises(ValueError):
        system.retrieve(" ")
    cases = load_golden(Path("data/golden_set.json"))
    result = evaluate(system, cases)
    assert len(result["cases"]) == 5
    assert 0 <= result["precision"] <= 0.2
    with pytest.raises(ValueError):
        evaluate(system, [{"pregunta": "x", "documentos_relevantes": ["missing"]}])


def test_pinecone_retriever_namespace_and_text():
    index = Mock()
    index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(metadata={"text": "contenido", "document_id": "a", "chunk_id": "1"})
        ]
    )
    embedder = Mock()
    embedder.embed_query.return_value = [0.1]
    retriever = PineconeRetriever(index=index, embedder=embedder, namespace="tenant-corpus")
    assert retriever.invoke("q")[0].page_content == "contenido"
    assert index.query.call_args.kwargs["namespace"] == "tenant-corpus"
    assert index.query.call_args.kwargs["include_metadata"] is True


def test_index_creation_validation(settings):
    client = Mock()
    client.list_indexes.return_value.names.return_value = []
    client.describe_index.return_value = SimpleNamespace(
        dimension=1536,
        metric="cosine",
        spec={"serverless": {"cloud": "aws"}},
        status={"ready": True},
        host="host",
    )
    ensure_index(settings, client)
    assert client.create_index.call_args.kwargs["dimension"] == 1536
    client.describe_index.return_value.dimension = 512
    with pytest.raises(ValueError):
        ensure_index(settings, client)
    client.describe_index.return_value.dimension = 1536
    client.describe_index.return_value.spec = {"pod": {}}
    with pytest.raises(ValueError):
        ensure_index(settings, client)


def test_ingest_publish_only_after_cloud_success(settings, monkeypatch):
    index = Mock()
    embedder = Mock()
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    monkeypatch.setattr("rag_system.cloud.ensure_index", lambda s: index)
    monkeypatch.setattr("rag_system.cloud.embeddings", lambda s: embedder)
    monkeypatch.setattr("rag_system.cloud.wait_visible", lambda *a: None)
    manifest = ingest(settings)
    batch = index.upsert.call_args.kwargs
    assert batch["namespace"] == manifest["namespace"]
    assert "text" in batch["vectors"][0]["metadata"]
    assert read_manifest(settings)[0] == manifest
    original = settings.manifest_path.read_bytes()
    index.upsert.side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError):
        ingest(settings)
    assert settings.manifest_path.read_bytes() == original


def test_visibility_timeout():
    index = Mock()
    index.fetch.return_value = SimpleNamespace(vectors={})
    with pytest.raises(TimeoutError):
        wait_visible(index, ["a"], "tenant", timeout=0)


def test_golden_validation(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text('[{"pregunta": "q", "documento_id_esperado": "a"}]')
    assert load_golden(path)[0]["documentos_relevantes"] == ["a"]
    path.write_text('[{"pregunta": "q"}]')
    with pytest.raises(ValueError):
        load_golden(path)


def test_long_document_token_chunking(settings, tmp_path):
    import tiktoken

    folder = tmp_path / "long"
    folder.mkdir()
    (folder / "long.md").write_text(
        "Documentación de automatización Python. " * 900, encoding="utf-8"
    )
    chunks = build_chunks(replace(settings, data_dir=folder))
    encoder = tiktoken.get_encoding("cl100k_base")
    assert len(chunks) > 1
    assert all(len(encoder.encode(doc.page_content)) <= 650 for doc in chunks)
    assert len({d.metadata["chunk_id"] for d in chunks}) == len(chunks)


def test_scanned_pdf_rejected(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with (tmp_path / "scanned.pdf").open("wb") as stream:
        writer.write(stream)
    with pytest.raises(ValueError, match="texto"):
        load_documents(tmp_path)
