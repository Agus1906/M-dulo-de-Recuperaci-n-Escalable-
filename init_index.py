from rag_system.cli import run
from rag_system.cloud import ensure_index
from rag_system.config import Settings

if __name__ == "__main__":
    run(lambda: (ensure_index(Settings.load()), print("Índice Serverless listo")))
