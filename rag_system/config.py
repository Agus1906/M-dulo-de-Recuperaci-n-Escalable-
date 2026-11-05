"""Configuration: environment secrets never enter manifests or logs."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    index_name: str
    namespace: str = "python-docs"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    cloud: str = "aws"
    region: str = "us-east-1"
    chunk_size: int = 650
    chunk_overlap: int = 100
    data_dir: Path = Path("data/documents")
    manifest_path: Path = Path("artifacts/corpus.json")

    def __post_init__(self):
        if not self.index_name.strip() or not self.namespace.strip():
            raise ValueError("INDEX_NAME y PINECONE_NAMESPACE son obligatorios")
        if self.model != "text-embedding-3-small" or self.dimension != 1536:
            raise ValueError("Esta entrega usa text-embedding-3-small con dimensión 1536")
        if not 0 <= self.chunk_overlap < self.chunk_size <= 800:
            raise ValueError("Requiere 0 <= CHUNK_OVERLAP < CHUNK_SIZE <= 800 tokens")

    @classmethod
    def load(cls, require_keys=True):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        if require_keys:
            missing = [k for k in ("PINECONE_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
            if missing:
                raise ValueError("Faltan variables: " + ", ".join(missing))
        return cls(
            index_name=os.getenv("INDEX_NAME", "preentrega4-rag"),
            namespace=os.getenv("PINECONE_NAMESPACE", "python-docs"),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "650")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            data_dir=Path(os.getenv("DATA_DIR", "data/documents")),
            manifest_path=Path(os.getenv("MANIFEST_PATH", "artifacts/corpus.json")),
        )
