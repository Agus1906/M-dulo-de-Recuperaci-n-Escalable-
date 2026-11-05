"""Source-level relevance, unique document IDs, macro averaged metrics."""

import json


def load_golden(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Golden Set vacío o inválido")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("pregunta"), str):
            raise ValueError("Cada caso requiere pregunta")
        expected = row.get("documentos_relevantes", [row.get("documento_id_esperado")])
        if (
            not row["pregunta"].strip()
            or not isinstance(expected, list)
            or not expected
            or not all(isinstance(x, str) and x.strip() for x in expected)
        ):
            raise ValueError("Pregunta o documentos relevantes inválidos")
        row["documentos_relevantes"] = sorted(set(expected))
    return rows


def metrics(retrieved, relevant, k=5):
    if k < 1 or not relevant:
        raise ValueError("k positivo y relevancias no vacías son obligatorios")
    top = list(dict.fromkeys(retrieved))[:k]
    hits = len(set(top) & set(relevant))
    return {"precision": hits / k, "recall": hits / len(set(relevant)), "hits": hits}


def evaluate(system, cases, k=5):
    known = {r["metadata"]["document_id"] for r in system.manifest["chunks"]}
    missing = {d for case in cases for d in case["documentos_relevantes"]} - known
    if missing:
        raise ValueError(
            "Golden Set referencia documentos inexistentes: " + ", ".join(sorted(missing))
        )
    results = []
    for case in cases:
        ids = [d.metadata["document_id"] for d in system.retrieve(case["pregunta"], k)]
        results.append(
            {
                "pregunta": case["pregunta"],
                "recuperados": ids,
                **metrics(ids, case["documentos_relevantes"], k),
            }
        )
    return {
        "k": k,
        "cases": results,
        "precision": sum(r["precision"] for r in results) / len(results),
        "recall": sum(r["recall"] for r in results) / len(results),
    }
