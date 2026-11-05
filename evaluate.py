import argparse
import json
from pathlib import Path

from rag_system.cli import run
from rag_system.config import Settings
from rag_system.evaluation import evaluate, load_golden
from rag_system.retrieval import LexicalBaseline, RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Precision@5 y Recall@5 por documento fuente")
    parser.add_argument("--golden", type=Path, default=Path("data/golden_set.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["hybrid", "lexical"], default="hybrid")
    args = parser.parse_args()
    cases = load_golden(args.golden)
    settings = Settings.load(require_keys=args.mode == "hybrid")
    system = RAGSystem(settings) if args.mode == "hybrid" else LexicalBaseline(settings)
    result = evaluate(system, cases)
    result["mode"] = args.mode
    print(f"Modo: {args.mode} (lexical = solo BM25 local, sin Pinecone/OpenAI)")
    for row in result["cases"]:
        print(
            f"{row['pregunta']}\n  {row['recuperados']}\n"
            f"  Precision@5={row['precision']:.3f} Recall@5={row['recall']:.3f}"
        )
    print(
        f"\nPromedio ({len(cases)} preguntas): Precision@5={result['precision']:.3f} "
        f"Recall@5={result['recall']:.3f}"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(main)
