# FINAL-SLIM-CORE-SYSTEM-GATE

Status kwalifikacji: `READY_FOR_REVIEW`  
Tryb: qualification only — bez zmian produkcyjnych, packaging, testów i skryptów.

## Zakres i walidowany stan

- validated HEAD: `5b700919563fb19df88896215cfc9211e5812246`
- `origin/development`: `5b700919563fb19df88896215cfc9211e5812246`
- required ancestor `b8623fc0886e28a8b1f39bac24c71b565142dbe0`: obecny
- branch: `development`
- platform: Windows `10.0.26200`
- Python: `3.12.11`
- uv: `0.8.15`
- preflight staging/worktree: czysty

Weryfikacja nie wykonywała optymalizacji zależności, migracji pakietów ani zmian runtime/RAG.

## Chronologia rozmiaru dependency footprint

| Punkt kontrolny | Dystrybucje | Site-packages | Harness import |
|---|---:|---:|---:|
| POST-LCI baseline | 204 | około 2.15 GB | około 54 s |
| DEP-2 | 179 | 764.75 MiB | 17.703 s (historyczny pomiar) |
| DEP-3 | 45 | 171.46 MiB | 1.875 s median |
| DEP-4 | 42 | 124.40 MiB | około 2.35 s |
| FINAL | 42 | 130,444,136 B / 124.40 MiB | 1.6744 s median |

Resolver podał dokładnie 42 dystrybucje w świeżym środowisku core:

```text
annotated-doc, annotated-types, anyio, boto3, botocore, certifi, cffi,
chardet, charset-normalizer, click, colorama, cryptography, distro, fastapi,
h11, httpcore, httpx, idna, intergrax-ai, jiter, jmespath, numpy, openai,
pycparser, pydantic, pydantic-core, python-dateutil, python-dotenv,
python-multipart, pyyaml, regex, requests, s3transfer, six, sniffio,
starlette, tiktoken, tqdm, typing-extensions, typing-inspection, urllib3,
uvicorn
```

## Clean core

### Deklaracje i instalacja

15 direct dependencies z `pyproject.toml`:

```text
fastapi, uvicorn, starlette, httpx, python-multipart, pydantic,
cryptography, python-dotenv, PyYAML, tqdm, openai, tiktoken, boto3,
numpy, chardet
```

Fresh Python 3.12 environment poza repozytorium, instalacja default `Intergrax-ai`:

```text
install: PASS
resolved: 42 distributions
site-packages: 130,444,136 B / 124.40 MiB
```

### Forbidden default proof

W default environment nie znaleziono żadnej z wymaganych rodzin:

```text
langchain*, langgraph*, torch, sentence-transformers, transformers,
openai-whisper, anthropic, mistralai, ollama, google-genai, cohere,
chromadb, qdrant-client, pinecone, beautifulsoup4, trafilatura,
python-docx, openpyxl, xlrd, pandas, pytesseract, pillow, PyMuPDF,
yt-dlp, webvtt-py, opencv-python-headless, streamlit, fastmcp, mcp
```

Łańcuchów tranzytywnych dla forbidden default nie stwierdzono. Core-owned modules sprawdzone osobno: załadowany został `tiktoken`; `openai` i `boto3` nie były ładowane przez lazy boundaries.

### Import i canonical smoke

Wymagane importy przeszły:

```text
import intergrax
import intergrax.runtime.nexus
import intergrax.harness
```

Harness import zmierzony w pięciu świeżych subprocessach:

```text
samples: 1.7107, 1.6575, 1.6765, 1.6591, 1.6744 s
min: 1.6575 s
median: 1.6744 s
max: 1.7107 s
```

Canonical contracts:

```text
KnowledgeDocument construction: PASS
KnowledgeDocument deterministic serialization: PASS
NexusLoop construction: PASS
HarnessApplication construction: PASS
LLM registry enumeration: PASS (19 registered providers)
embedding registry enumeration without provider construction: PASS (default registry: hf)
in-memory vector manager/path: PASS
native text handler construction: PASS
native text ingest fixture: PASS in targeted suite
configuration/contracts imports: PASS
```

Po canonical smoke `sys.modules` nie zawierał żadnego z forbidden modules. `torch`, `sentence_transformers`, `transformers`, provider SDKs, vector SDKs, `streamlit`, `fastmcp` i `mcp` pozostały niezaładowane.

## LCI i LangChain/LangGraph

```text
validate_langchain_inventory.py: PASS
  69 unique inventory IDs
  0 duplicate path + line + symbol
  0 unclassified
  summary/package totals match

check_langchain_boundary.py: PASS
  4442 production files scanned
  10 allowed-zone imports
  5 grandfathered guarded imports
  0 new forbidden imports
  0 stale grandfather entries
```

Default core:

```text
installed langchain*: 0
installed langgraph*: 0
LangChain required by canonical runtime: NO
LangGraph required by canonical runtime: NO
```

## Dependency governance

```text
check_dependency_ownership.py: PASS
core direct dependencies: 15
optional direct declarations: 102
```

Negatywne próby na tymczasowych kopiach `pyproject.toml` — repozytoryjny plik nie był modyfikowany:

```text
forbidden core: FAIL as expected
  CORE_OWNERSHIP_VIOLATION, FORBIDDEN_CORE_PACKAGE,
  DUPLICATE_CORE_EXTRA_OWNERSHIP

LangChain core leak: FAIL as expected
  CORE_OWNERSHIP_VIOLATION, LANGCHAIN_CORE_LEAK,
  DUPLICATE_CORE_EXTRA_OWNERSHIP

unbounded major (pydantic>=2.7): FAIL as expected
  UNBOUNDED_MAJOR

direct mcp: FAIL as expected
  CORE_OWNERSHIP_VIOLATION, FORBIDDEN_CORE_PACKAGE,
  PROHIBITED_DIRECT_TRANSITIVE
```

## Optional capability matrix

Każdy przypadek był instalowany w świeżym środowisku. Kolumna `D` oznacza liczbę resolved distributions; wymagane moduły były obecne i importowalne.

### Individual extras

| Extra | Install/import | D | LangChain/LangGraph |
|---|---|---:|---|
| `llm-openai` | PASS | 42 | 0 |
| `llm-anthropic` | PASS | 44 | 0 |
| `llm-mistral` | PASS | 53 | 0 |
| `llm-ollama` | PASS | 43 | 0 |
| `llm-gemini` | PASS | 48 | 0 |
| `llm-bedrock` | PASS | 42 | 0 |
| `llm-cohere-native` | PASS | 51 | 0 |
| `rag-local-embeddings` | PASS | 61 | 0 |
| `vector-chroma` | PASS | 98 | 0 |
| `vector-qdrant` | PASS | 50 | 0 |
| `vector-pinecone` | PASS | 46 | 0 |
| `parsing-web` | PASS | 56 | 0 |
| `parsing-office` | PASS | 79 | explicit LangChain family |
| `parsing-pdf` | PASS | 71 | explicit LangChain family |
| `parsing-ocr` | PASS | 45 | 0 |
| `media-whisper` | PASS | 56 | 0 |
| `media-youtube` | PASS | 43 | 0 |
| `media-video` | PASS | 44 | 0 |
| `media-image` | PASS | 43 | 0 |
| `ui-streamlit` | PASS | 66 | 0 |
| `mcp` | PASS | 90 | 0 |

`parsing-office` i `parsing-pdf` jawnie deklarują `langchain-community`; wynikowe `langchain-classic`, `langchain-core`, `langchain-protocol` i `langchain-text-splitters` są oczekiwane wyłącznie w tych capability environments.

### Compatibility extras

| Extra | Install/import | D | Result |
|---|---|---:|---|
| `rag-langchain-loaders` | PASS | 70 | expected compatibility boundary |
| `rag-langchain-embeddings` | PASS | 57 | expected compatibility boundary |
| `rag-langchain-splitters` | PASS | 56 | expected compatibility boundary |
| `llm-langchain-ollama` | PASS | 57 | expected compatibility boundary |
| `langgraph-legacy` | PASS | 60 | expected legacy boundary |

### Aggregate i kombinacje

| Environment | Install/import | D | LangChain/LangGraph |
|---|---|---:|---|
| `llm-all` | PASS | 71 | 0 |
| `llm-all + rag-local-embeddings` | PASS | 84 | 0 |
| `vector-qdrant + rag-local-embeddings` | PASS | 69 | 0 |
| `vector-chroma + parsing-office` | PASS | 123 | explicit parsing-office family |
| `parsing-office + parsing-pdf` | PASS | 80 | explicit parsing family |
| `media-whisper + media-youtube` | PASS | 57 | 0 |
| `mcp + harness-author` | PASS | 90 | 0 |

Świeże fixture environments:

```text
llm-all provider constructors: PASS
vector constructors (Chroma/Qdrant/Pinecone): PASS
parsing constructors/fixtures: PASS
media/UI/MCP constructors/fixtures: PASS
```

Nie wykonywano żadnych płatnych API calls ani model downloads.

## Runtime qualification

### Targeted mandatory selection

Wynik:

```text
226 passed, 2 failed, 6 warnings
```

Failures zostały sklasyfikowane:

1. `test_negative_proof_new_import_detected` — `KNOWN_UNRELATED`; testowy subprocess checker zwrócił błąd na stderr, a asercja oczekuje znacznika na stdout. Nie jest to regresja dependency modernization.
2. `test_docling_strategy_uses_private_handle_and_skips_empty_items` — `ENVIRONMENT`; `docling_core` nie jest instalowane w default/test groups.

### Native LLM i Ollama

```text
native provider registry / adapter / structured output / tools /
usage / conformance targeted selection: PASS poza findings powyżej
native Ollama live qualification: 1 passed
live model: qwen2.5:7b
Ollama model pull: NIE
```

Deterministyczna live qualification działała z istniejącym lokalnym serwerem Ollama; `ollama list` wykazało dostępne modele.

### Knowledge/RAG

```text
KnowledgeDocument conformance: PASS
RAG tenant/vector bounded selection: 33 passed, 1 skipped
explicit tenant_id selection: 22 passed
```

### Parser/media/MCP guards

```text
73 passed, 2 failed, 2 skipped
```

Failures:

- DOCX fixture: brak `docx2txt` w default/test environment,
- HTML fixture: brak `unstructured` w default/test environment.

Obie biblioteki są opcjonalne i nie są częścią default core; capability installs oraz constructor fixtures przeszły. Findings pozostają udokumentowane do osobnego review-fix.

## Known drift i klasyfikacja

### Provider runtime cutover

Historyczny DEP-4 wynik: `18` pre-existing failures. Aktualne bounded rerun:

```text
1522 passed, 16 failed
```

Failures nadal dotyczą tego samego niezależnego obszaru legacy provider cutover (brak legacy factory dla trzech kanałów, MongoDB fake/index contract oraz client/signature/import assumptions). Dependency modernization relation: `NO`. Nie zmieniano tych testów ani provider runtime.

### RAG tenant drift

W aktualnym bounded tenant selection: `present = NO` dla tego selection (`33 passed, 1 skipped`). Szerszy run ujawnił jeden znany przypadek `explicit tenant_id` fallback (`create_vectorstore_from_integration`), więc globalnie:

```text
present: YES
owner: RAG stream
dependency modernization relation: NO
```

Pozostałe RAG failures w broad run dotyczą istniejącego abstract fake vectorstore interface drift (`list_source_record_ids`) i są niezależne od dependency modernization.

## Broader confidence run

Polecenie objęło non-network selection dla LLM adapters, RAG, architecture, vector store, document parser i MCP:

```text
719 passed
12 failed
2 skipped
211 deselected
```

Klasyfikacja wszystkich failures:

```text
DEPENDENCY_REGRESSION: 0
KNOWN_UNRELATED: 11
ENVIRONMENT: 1
NEW_UNRELATED: 0
```

Wynik nie był traktowany jako pełny repository PASS; znane drift issues pozostają poza zakresem tego gate.

## Package health

```text
uv lock --check: PASS
fresh core uv pip check: PASS (42 packages compatible)
workspace uv pip check: PASS (260 packages compatible)
```

## Independent verdicts

```text
CORE MINIMALITY: PASS
CORE FUNCTIONALITY: PASS
OPTIONALIZATION: PASS_WITH_FINDINGS
LLM PROVIDERS: PASS
LOCAL ML: PASS
VECTOR CAPABILITIES: PASS
PARSING: PASS_WITH_FINDINGS
MEDIA: PASS
UI/MCP: PASS
LANGCHAIN INDEPENDENCE: PASS
LANGGRAPH ISOLATION: PASS
VERSION POLICY: PASS
GOVERNANCE: PASS
RESOLVER HEALTH: PASS
RUNTIME REGRESSION: PASS_WITH_FINDINGS
```

Final qualification verdict: `READY_FOR_REVIEW`.

Warunek review acceptance jest spełniony: clean core i canonical smoke przechodzą, default nie zawiera niezamierzonych optional dependencies, LangChain/LangGraph są nieobecne w core, governance i negatywne probes przechodzą, extras i high-value combinations rozwiązują się, lock/pip check przechodzą, a `DEPENDENCY_REGRESSION = 0`. Znane provider/RAG/parser environment findings wymagają osobnego review-fix i nie zostały naprawione.

## Evidence commands

```text
uv run python scripts/docs/validate_langchain_inventory.py
uv run python scripts/maintenance/check_langchain_boundary.py
uv run python scripts/maintenance/check_dependency_ownership.py
uv lock --check
uv pip check
```

Targeted i broad pytest selections są zapisane w qualification run evidence; wszystkie findings powyżej mają klasyfikację.

## Change budget

```text
production changes: 0
packaging changes: 0
test changes: 0
script changes: 0
qualification documents: 1
```

Roadmap:

```text
DEP-1: APPROVED
DEP-2: APPROVED
DEP-3: APPROVED
DEP-4: APPROVED
FINAL SLIM-CORE GATE: READY_FOR_REVIEW
```
