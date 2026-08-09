# LCI-7C — LangChain compatibility installation gate

## Verdict

**COMPATIBILITY_PACKAGING_DEFECT**

Validated checkout SHA: `508b86bc49d227cc7ea135ab63c9cf1ef299273c`
Platform: Windows 10 (`win32`)
Python: 3.12.11
Validated: 2026-08-09

The clean-core control and three compatibility families passed. The loader and
splitter families do not qualify with the current resolver result. No
production or RAG files were changed, and LCI-7D was not started.

## Controls

- 7B clean-core installation: **PASS**
- Installed package origin: temporary environment `site-packages`, not checkout
- `PYTHONNOUSERSITE=1`: **PASS**
- Core `[project].dependencies`: zero LangChain/LangGraph distributions
- Missing-extra control (Ollama and representative RAG embedding provider): **PASS**
- Each extra was installed in its own temporary Python 3.12 environment outside the repository; no shared `.venv` was used.

## Compatibility families

| Extra | Direct distributions | Installed compatibility closure | Result |
|---|---|---|---|
| `llm-langchain-ollama` | `langchain-core`, `langchain-ollama` | `langchain-core`, `langchain-ollama`, `langchain-protocol` | PASS |
| `rag-langchain-loaders` | `langchain-community` | `langchain-classic`, `langchain-community`, `langchain-core`, `langchain-protocol`, `langchain-text-splitters` | FAIL |
| `rag-langchain-embeddings` | `langchain-ollama` | `langchain-core`, `langchain-ollama`, `langchain-protocol` | PASS |
| `rag-langchain-splitters` | `langchain-text-splitters` | `langchain-core`, `langchain-protocol`, `langchain-text-splitters` | FAIL |
| `langgraph-legacy` | `langgraph` | `langchain-core`, `langchain-protocol`, `langgraph`, `langgraph-checkpoint`, `langgraph-prebuilt`, `langgraph-sdk` | PASS |

LLM compatibility passed import, construction, deterministic plain, tools,
structured, stream, and native `LLMAdapterRegistry.create(OLLAMA)` ownership
checks. It returned `NativeOllamaAdapter`, not the compatibility adapter.

Embeddings passed deterministic mocked `OllamaEmbeddings` construction,
dimension, and vector ABI checks. LangGraph was limited to distribution and
legacy module import boundary; no legacy supervisor workflow was run.

## Failure evidence

`rag-langchain-loaders` failed when resolving `UnstructuredHTMLLoader`: the
installed `langchain-community` path imports `langchain-text-splitters`, then
`sentence-transformers`/`transformers`, which raises `NameError: name 'torch' is
not defined` because the core installation pins `torch==2.2.2+cpu` while that
transitive stack expects a newer Torch surface.

`rag-langchain-splitters` reaches the same transitive
`sentence-transformers`/`transformers` and pinned-Torch conflict while creating
the canonical native default splitter. The explicit compatibility strategy
module itself imports, but the family cannot prove native-default ownership
under the resolved installation.

This is reported without modifying the optional dependency declarations,
provider implementation, registry, or RAG code.

## Tests and audits

- Existing targeted compatibility tests: **20 passed**
- `validate_langchain_inventory.py`: **PASS**
- `check_langchain_boundary.py`: **PASS**
- `check_langgraph_not_required.py`: **FAIL** on existing generated runtime-context findings only; not repaired in 7C
- `uv lock --check`: **PASS**
- Ruff: **PASS**
- Pyright: **PASS after gate-only annotation fix**
- `py_compile`: **PASS**
- `git diff --check`: **PASS** (line-ending warnings only)

## Changed files

- `scripts/ci/check_langchain_compatibility_install.py`
- `.github/workflows/unit-tests.yml`
- this evidence file

LCI-7A: **APPROVED**
LCI-7B: **APPROVED**
LCI-7C: **NOT READY — COMPATIBILITY_PACKAGING_DEFECT**
LCI-7D: **NEXT AFTER ACCEPTANCE; NOT STARTED**
