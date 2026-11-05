import argparse

from rag_system.cli import run
from rag_system.config import Settings
from rag_system.retrieval import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Recuperación híbrida BM25 + Pinecone")
    parser.add_argument("query")
    args = parser.parse_args()
    docs = RAGSystem(Settings.load()).retrieve(args.query)
    for i, doc in enumerate(docs, 1):
        print(
            f"{i}. {doc.metadata['document_id']} | {doc.metadata['source']} "
            f"| página {doc.metadata['page']}\n{doc.page_content}\n"
        )


if __name__ == "__main__":
    run(main)
