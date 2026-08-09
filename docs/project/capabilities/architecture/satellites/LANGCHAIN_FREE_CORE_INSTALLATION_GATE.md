# LangChain-free core installation gate

## Validation receipt

- Validated SHA: `a3f02282508e69bb624bc5eeb235b440f3d24d4a`
- Date: 2026-08-09
- Platform: Windows 10.0.26200 (`win32`)
- Python: 3.12.11
- Installation profile: default project package, no optional extras and no dev dependency group
- Installation command: `uv pip install --python <temporary-env>\Scripts\python.exe .`
- Installed distributions: 205
- Shared repository `.venv` modified: no

## Clean-install result

The temporary environment was created outside the repository and removed after the
proof. The package was installed from the current checkout, not from PyPI.

| Distribution family | Result |
|---|---|
| `langchain` | NOT INSTALLED |
| `langchain-core` | NOT INSTALLED |
| `langchain-community` | NOT INSTALLED |
| `langchain-openai` | NOT INSTALLED |
| `langchain-ollama` | NOT INSTALLED |
| `langchain-text-splitters` | NOT INSTALLED |
| `langgraph` | NOT INSTALLED |

The gate also rechecked all installed distribution names after runtime smoke and
found no `langchain*` or `langgraph*` distribution. `PYTHONNOUSERSITE=1` and a
controlled import blocker prevent user-site or `PYTHONPATH` availability from
masking the result.

## Native/core qualification

- `[project].dependencies` contains zero `langchain*` and `langgraph*` entries: PASS
- `intergrax`, `intergrax.llm.messages`, `intergrax.llm_adapters.contracts`,
  `intergrax.knowledge.contracts`: PASS
- `LLMAdapterRegistry` / `LLMProvider` registered-provider listing: PASS
- `LLMAdapterRegistry.create(LLMProvider.OLLAMA, ...)` returned
  `NativeOllamaAdapter`: PASS
- Native Ollama deterministic response and usage ABI with an injected fake
  client: PASS
- Native OpenAI module/class import without credentials or network: PASS
- `KnowledgeDocument` tenant, optional namespace/workspace semantics, identity
  and content: PASS
- Nexus/Harness `InMemoryArtifactStore` construction and round-trip: PASS
- CLI/core entrypoint: `NOT_APPLICABLE` for this offline core gate

## Read-only native RAG consumer smoke

- RAG source: current validated SHA above
- Canonical path: `KnowledgeDocument` -> `RecursiveChunkingStrategy` ->
  deterministic embedding fixture -> native `InMemoryVectorStore` ->
  `VectorstoreManager` -> `VectorSimilarityRetriever` ->
  `RetrievalService`
- Known document retrieval: PASS
- RAG-owned files changed: none
- Compatibility providers, LangChain splitters, LangGraph adapters, external
  vector stores, reranking, Graph RAG and live services were intentionally
  excluded.

## Validation checks

- Targeted normal-environment regression: `103 passed`, 1 existing Pydantic warning
- `uv lock --check`: PASS
- `git diff --check`: PASS
- Ruff, Pyright and `py_compile` for the gate script: PASS
- `check_langchain_boundary.py`: PASS
- `check_langgraph_not_required.py`: reports four pre-existing imports in generated
  runtime-context copies; no files in this task were changed to mask or repair them

## Final verdict

`PASS`

- LCI-7A: `APPROVED`
- LCI-7B: `READY_FOR_REVIEW`
- LCI-7C: `NEXT AFTER ACCEPTANCE`
