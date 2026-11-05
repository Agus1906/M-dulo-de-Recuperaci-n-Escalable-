# Validación comprobada · 5 de noviembre de 2026

Entorno: Windows, Python 3.12.15. Versiones exactas: `requirements-lock.txt`.

| Verificación | Resultado |
| --- | --- |
| pytest, 17 casos | 17 aprobados |
| Ruff | Sin errores |
| Compilación de módulos y scripts | Sin errores |
| pip check | Sin dependencias incompatibles |
| Falta de credenciales | init_index.py termina con código 1 y enumera variables faltantes |
| Benchmark BM25 local, 5 preguntas | Precision@5 = 0,200; Recall@5 = 1,000 |
| Inicialización/ingesta/recuperación Pinecone real | Pendiente: no hay credenciales |
| Evaluación híbrida BM25 + Pinecone real | Pendiente: no hay credenciales |

## Reproducir

```powershell
python -m pytest -q
python -m ruff check .
python -m compileall -q rag_system init_index.py ingest.py query.py evaluate.py
python -m pip check
python evaluate.py --mode lexical --output artifacts/evaluation-lexical.json
```

En esta máquina se usó `--basetemp=artifacts/test-run-04` porque el directorio temporal global de pytest tiene restricciones de acceso. La primera ejecución del tokenizador requirió conexión para descargar `cl100k_base`.

El resultado local está en `evaluation/bm25-local.json`. Las cinco fuentes esperadas aparecieron en Top-5 (también fueron Top-1 en esta ejecución). No son métricas del retriever híbrido. Hay una sola fuente relevante por pregunta, por lo que Precision@5 no puede superar 0,20.

Los tests cubren validación de configuración, corpus determinista, metadata, integridad del manifest, aislamiento de namespaces, carga Markdown, JSON inválido, PDF sin texto, chunking con límite real de tokens, fórmulas de precisión/recall, deduplicación de fuentes, fusión con retriever semántico simulado, creación/validación de índice, contratos de query, publicación del manifest después de éxito, conservación ante fallos, timeout de visibilidad y validación del Golden Set.

La advertencia de deprecación de `langchain-community` se conserva visible: se usa su BM25Retriever por requerimiento de la actividad. Los dobles cloud no demuestran funcionamiento de servicios reales.

## Verificación cloud pendiente

1. Completar `.env` con claves propias de Pinecone y OpenAI.
2. Ejecutar `python init_index.py` y verificar Serverless, dimensión 1536 y métrica cosine.
3. Ejecutar `python ingest.py` y comprobar el namespace mostrado y metadata en la consola Pinecone.
4. Ejecutar `python query.py "¿Cómo usar placeholders en sqlite3?"`.
5. Ejecutar `python evaluate.py --output artifacts/evaluation-hybrid.json`.
6. Repetir ingesta: mismo namespace e IDs, sin aumento de conteo por duplicación.
7. Actualizar README y este registro con métricas híbridas observadas. No reemplazar el rótulo del benchmark local para presentarlo como híbrido.
