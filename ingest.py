from rag_system.cli import run
from rag_system.cloud import ingest
from rag_system.config import Settings

if __name__ == "__main__":
    run(lambda: ingest(Settings.load()))
