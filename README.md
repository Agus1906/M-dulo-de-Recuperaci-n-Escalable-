# Pre-entrega 4 · Módulo de Recuperación Escalable

Servicio de recuperación en Python: **Pinecone Serverless + embeddings OpenAI + BM25**, fusionados mediante `EnsembleRetriever` de LangChain. Devuelve los **cinco documentos fuente** mejor clasificados, con contenido, fuente, página y categorías. No necesita una base relacional para recuperar texto.

## Estado de validación

- Implementación y pruebas locales incluidas; consultar `VALIDATION.md` para resultados comprobados.
- **Evaluación híbrida en la nube pendiente**: no se dispone de claves Pinecone/OpenAI. No se inventan métricas ni se presentan los mocks como una ejecución cloud.
- `evaluate.py --mode lexical` permite medir una referencia BM25 real y local, sin credenciales. No valida la recuperación vectorial.

## 1. Instalación

Requiere **Python 3.12**. La primera ejecución de chunking descarga el tokenizador público `cl100k_base` y necesita conexión; luego utiliza su caché. Ejecutar desde la raíz de este repositorio (rutas relativas a ese directorio).

```powershell
git clone https://github.com/Agus1906/M-dulo-de-Recuperaci-n-Escalable-.git
cd M-dulo-de-Recuperaci-n-Escalable-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
# Para reproducir las versiones exactas verificadas:
# python -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

En Linux/macOS: `source .venv/bin/activate` y `cp .env.example .env`.

## 2. Configuración de cuentas e índice

Crear una API key en la consola de Pinecone y otra en OpenAI con acceso a embeddings y saldo disponible. Completar **localmente** `.env`:

```dotenv
PINECONE_API_KEY=tu-clave-pinecone
OPENAI_API_KEY=tu-clave-openai
INDEX_NAME=preentrega4-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=python-docs
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
CHUNK_SIZE=650
CHUNK_OVERLAP=100
DATA_DIR=data/documents
MANIFEST_PATH=artifacts/corpus.json
```

`.env` está excluido de Git. Nunca compartir claves en commits ni logs. No se reutilizan claves de otros proyectos. Anthropic no es un proveedor de embeddings intercambiable en esta implementación.

```powershell
python init_index.py
```

El script crea el índice si falta, con **1536 dimensiones, cosine y Serverless**, en la nube/región configuradas. Si ya existe, valida dimensión, métrica y tipo; no borra ni recrea índices incompatibles. Espera su disponibilidad con plazo de 120 segundos. Si el plan no permite esa región, ajustar la configuración antes de crear un índice nuevo.

## 3. Ingesta

```powershell
python ingest.py
```

1. Lee recursivamente PDF, Markdown y JSON desde `DATA_DIR` y valida sus campos.
2. Divide con `RecursiveCharacterTextSplitter`, midiendo en tokens `cl100k_base`: máximo 650, solapamiento 100. Se verifica la longitud real de cada salida; si el conteo por separadores excede el límite, el fragmento se subdivide por caracteres Unicode sin perder contenido (ese corte adicional no agrega solapamiento). Los documentos breves se conservan completos; no se rellenan artificialmente hasta 500 tokens.
3. Genera embeddings `text-embedding-3-small`, dimensión 1536, en lotes de 64.
4. Hace upsert con identificadores SHA-256 deterministas y metadatos `text`, `document_id`, `chunk_id`, `source`, `page`, `categories`.
5. Comprueba visibilidad de los IDs y publica atómicamente `artifacts/corpus.json`, que conserva exactamente los chunks necesarios para BM25.

El namespace efectivo es `<PINECONE_NAMESPACE>-<hash del corpus/modelo>`. Cambiar documentos crea una generación aislada: no mezcla vectores anteriores con BM25 nuevo. Repetir una ingesta del mismo corpus sobrescribe los mismos IDs; no duplica vectores. Un fallo no reemplaza el manifest anterior. Los namespaces viejos se conservan y pueden consumir almacenamiento: su limpieza requiere una decisión explícita; no hay borrados automáticos. Ejecutar un solo proceso de ingesta por manifest.

El índice es eventualmente consistente. La comprobación por fetch reduce el riesgo de consulta inmediata incompleta, pero no garantiza que el ranking de búsqueda ya esté actualizado; si faltan resultados después de la ingesta, esperar y repetir la evaluación.

### Dataset incluido

`data/documents/python.json` contiene **ocho fichas técnicas originales en español** sobre la biblioteca estándar Python, con referencias oficiales; no es un volcado de documentación externa. Temas: pathlib, JSON, logging, asyncio, SQLite, unittest, CSV y datetime. Cada registro tiene un ID estable.

Formato para incorporar JSON:

```json
[
  {
    "document_id": "manual-transacciones",
    "text": "Contenido original del documento...",
    "page": 1,
    "categories": ["base-de-datos", "python"]
  }
]
```

En PDF se conserva el número de página real. Markdown y JSON sin página usan 1 como página lógica. PDF escaneado requiere OCR previo; no se implementa OCR. Distintos archivos no pueden compartir un `document_id`.

## 4. Recuperación híbrida

```powershell
python query.py "¿Cómo limitar operaciones concurrentes con asyncio.Semaphore?"
```

`RAGSystem` encapsula un `EnsembleRetriever` con pesos **0,5 BM25 / 0,5 vectorial** y fusión por Reciprocal Rank Fusion (RRF). No suma directamente scores de escalas distintas. Cada rama busca hasta 20 chunks; la fusión deduplica por `chunk_id` y la salida por `document_id`, conservando el fragmento mejor posicionado de cada fuente. Devuelve hasta cinco fuentes: si hay menos de cinco candidatas únicas, devuelve menos.

BM25 normaliza mayúsculas y tokeniza identificadores técnicos. El retriever vectorial usa el SDK nativo de Pinecone y devuelve texto desde metadata. El mismo namespace se usa en ingesta, fetch y query. `read_manifest` comprueba integridad y compatibilidad de configuración antes de construir BM25.

```python
from rag_system.config import Settings
from rag_system.retrieval import RAGSystem

rag = RAGSystem(Settings.load())
for document in rag.retrieve("¿Cómo usar placeholders en sqlite3?"):
    print(document.metadata, document.page_content)
```

## 5. Golden Set y métricas

`data/golden_set.json` contiene cinco preguntas, cada una con una fuente conocida, definida antes de ejecutar la búsqueda. `documentos_relevantes` permite más de una fuente válida; también se admite el formato mínimo `documento_id_esperado`.

```powershell
# Evaluación solicitada: Pinecone + BM25, después de ingest.py
python evaluate.py --output artifacts/evaluation-hybrid.json

# Referencia local independiente, sin servicios cloud
python evaluate.py --mode lexical --output artifacts/evaluation-lexical.json
```

- **Precision@5** = documentos fuente relevantes únicos recuperados / 5.
- **Recall@5** = documentos fuente relevantes únicos recuperados / total de fuentes relevantes de la pregunta.
- El resumen es el promedio aritmético de las cinco preguntas; una salida corta mantiene denominador 5 en precisión.
- Con una única fuente relevante por pregunta, acertar produce Precision@5 = **0,20** y Recall@5 = **1,00**. El máximo de precisión de este Golden Set es 0,20, no 1,00.
- La relevancia se juzga a nivel de fuente según el Golden Set; no mide calidad de una respuesta generada ni utilidad de cada frase del chunk. El benchmark pequeño es una comprobación funcional y debe ampliarse para estimar calidad real.

**Resumen verificado (solo BM25 local, 5 preguntas): Precision@5 = 0,200; Recall@5 = 1,000.** La evaluación híbrida cloud está pendiente de credenciales. Ver `VALIDATION.md` para alcance y evidencia.

## 6. Pruebas y validación antes de entregar

```powershell
python -m pytest -q
python -m ruff check .
python -m compileall -q rag_system init_index.py ingest.py query.py evaluate.py
```

Las pruebas usan dobles de servicios cloud: validan lógica y contratos esperados, no credenciales ni compatibilidad real del entorno remoto. Antes de considerar completa la validación cloud: ejecutar inicialización, ingesta, consulta y evaluación; comprobar metadata en Pinecone; repetir ingesta y verificar que el conteo del mismo namespace no aumenta; guardar el JSON de métricas y actualizar el resumen con resultados observados.

## Arquitectura y límites operativos

```text
data/documents → carga/validación → chunks (tokens) → OpenAI embeddings
                                                     ↓
                                            Pinecone Serverless
                                                     ↓
corpus.json → BM25Retriever ───────┐     PineconeRetriever
                                 └───────┬───────────┘
                                  EnsembleRetriever (RRF)
                                           ↓
                             Top-5 fuentes → evaluate.py
```

- `rag_system/config.py`: configuración y validaciones.
- `documents.py`: entrada, normalización, chunking y snapshots.
- `cloud.py`: infraestructura, embeddings, lotes, reintentos y publicación.
- `retrieval.py`: recuperación híbrida y baseline local.
- `evaluation.py`: Golden Set y cálculo de métricas.
- `cli.py`: logs y salida no cero ante errores.
- `.github/workflows/tests.yml`: pruebas automáticas sin secretos.

Se fija Pinecone SDK **7.3.0** para usar la API `create_index`/`ServerlessSpec`; no actualizar al SDK mayor actual sin revisar sus cambios. `langchain-community` emite una advertencia de deprecación; se conserva aquí porque la consigna pide su `BM25Retriever`. Las versiones exactas instaladas se conservan en `requirements-lock.txt`. Referencias: [Pinecone Python](https://sdk.pinecone.io/python/), [EnsembleRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever).

Pinecone escala la rama vectorial. **BM25 sigue siendo local y residente en memoria**: este prototipo no implica escalabilidad distribuida del componente léxico. Para un corpus grande se recomienda un motor léxico externo o sparse vectors, conservando aislamiento por inquilino y evaluación de relevancia. El namespace es partición de datos, no autorización; no hay API pública ni autenticación multiusuario en esta entrega.

La ejecución es manual por CLI. Si la computadora se apaga, la ingesta se interrumpe; al reejecutarla se reconstruyen embeddings y se repiten upserts idempotentes. No hay checkpoint de embeddings ni garantía de evitar repetir su costo. Reintentos acotados para fallos transitorios de upsert y del cliente OpenAI; errores de credenciales/validación no se reintentan. Logs por consola con lotes y namespace, sin texto ni claves; los errores de SDK se resumen por tipo para no exponer cuerpos sensibles. No hay notificaciones ni scheduler: para producción ejecutar en un runner persistente y configurar alertas. No incluye un generador LLM: el alcance obligatorio es el módulo de recuperación y su evaluación.
