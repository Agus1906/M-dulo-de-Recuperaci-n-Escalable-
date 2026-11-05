"""Hybrid RRF retrieval. BM25 is a local snapshot; semantic vectors live in Pinecone."""

import re
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .cloud import embeddings, ensure_index
from .documents import read_manifest


def tokenize(text):
    return re.findall(r"[\w]+", text.casefold())


class PineconeRetriever(BaseRetriever):
    index: Any
    embedder: Any
    namespace: str
    candidates: int = 20

    def _get_relevant_documents(self, query, *, run_manager):
        response = self.index.query(
            vector=self.embedder.embed_query(query),
            top_k=self.candidates,
            namespace=self.namespace,
            include_metadata=True,
        )
        return [
            Document(
                page_content=m.metadata["text"],
                metadata={k: v for k, v in m.metadata.items() if k != "text"},
            )
            for m in response.matches
        ]


class RAGSystem:
    def __init__(self, settings, *, semantic=None):
        self.manifest, docs = read_manifest(settings)
        lexical = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
        lexical.k = min(20, len(docs))
        semantic = (
            semantic
            if semantic is not None
            else PineconeRetriever(
                index=ensure_index(settings),
                embedder=embeddings(settings),
                namespace=self.manifest["namespace"],
                candidates=min(20, len(docs)),
            )
        )
        self.retriever = EnsembleRetriever(
            retrievers=[lexical, semantic],
            weights=[0.5, 0.5],
            id_key="chunk_id",
        )

    def retrieve(self, query, k=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("La consulta no puede estar vacía")
        if not 1 <= k <= 20:
            raise ValueError("k debe estar entre 1 y 20")
        unique = {}
        for doc in self.retriever.invoke(query):
            unique.setdefault(doc.metadata["document_id"], doc)
        return list(unique.values())[:k]


class LexicalBaseline:
    """Offline BM25 diagnostic, never a replacement for the cloud hybrid benchmark."""

    def __init__(self, settings):
        from .documents import build_chunks, make_manifest

        docs = build_chunks(settings)
        self.manifest = make_manifest(settings, docs)
        self.retriever = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
        self.retriever.k = min(20, len(docs))

    def retrieve(self, query, k=5):
        unique = {}
        for doc in self.retriever.invoke(query):
            unique.setdefault(doc.metadata["document_id"], doc)
        return list(unique.values())[:k]
