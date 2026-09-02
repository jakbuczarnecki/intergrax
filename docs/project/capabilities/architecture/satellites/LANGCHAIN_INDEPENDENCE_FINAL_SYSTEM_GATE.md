# LangChain Independence Final System Gate

**Task:** `LCI-FINAL-SYSTEM-GATE-POST-LANGCHAIN-INDEPENDENCE`
**Date:** 2026-08-09
**Platform:** Windows 10 (`win32`)
**Python:** 3.12.11
**Validated HEAD:** `3b36ac38ce5693da1570172482285d0dad159fe3`
**origin/development:** `3b36ac38ce5693da1570172482285d0dad159fe3`
**Starting HEAD:** `079123ab1f715c95ccba8bd3e5979491010586d3`
**Later parallel commit:** `3b36ac38 fix(runtime): recover shared E2E dependency availability`

All required ancestors (`079123ab`, `6e6a446e`, `20bc04d`) are ancestors of the validated HEAD. The working tree contained unrelated concurrent unstaged work; the index was empty at the gate checkpoints. No production, packaging, or RAG files were changed by this gate.

## Results

### LCI-7B clean-core installation

**PASS.** The canonical isolated install gate installed 205 distributions, with zero installed `langchain*` or `langgraph*` distributions. Core imports, registry, native Ollama, native OpenAI, `KnowledgeDocument`, minimal native RAG, and Nexus/Harness artifact round-trip all passed. The missing-extra control also passed.

### LCI-7C compatibility installation

**FAIL - `FINAL_GATE_COMPATIBILITY_FAILURE`.**

| Compatibility family | Result |
|---|---|
| `llm-langchain-ollama` | PASS |
| `rag-langchain-loaders` | FAIL: `NameError: name 'torch' is not defined` while importing `langchain_community` |
| `rag-langchain-embeddings` | PASS |
| `rag-langchain-splitters` | FAIL: `NameError: name 'torch' is not defined` |
| `langgraph-legacy` | PASS |

The failures occurred in fresh isolated Python 3.12 environments. No production, RAG, or compatibility implementation was modified to address them.

## Core and native contracts

- **Core packaging:** `[project].dependencies` contains zero LangChain and zero LangGraph entries.
- **Compatibility extras:** `llm-langchain-ollama`, `rag-langchain-loaders`, `rag-langchain-embeddings`, `rag-langchain-splitters`, and `langgraph-legacy` remain declared.
- **Native Ollama:** registry default resolves to `NativeOllamaAdapter`. Deterministic plain, tools, structured, stream, provider usage counters, usage fallback, and capability/context regression coverage passed.
- **Native OpenAI:** module/class import and registry/default contract passed; no `langchain-openai` dependency is required.
- **`KnowledgeDocument`:** tenant, namespace, optional workspace semantics, identity, content, metadata, and provenance contract passed.
- **Native RAG read-only:** `KnowledgeDocument` → native splitter/chunker → deterministic embedding fixture → in-memory vector → retrieval passed.
- **Nexus/Harness:** in-memory artifact creation, storage, lookup, and run-list round-trip passed.

## Audits and lock

- **Inventory:** PASS - 69 unique inventory IDs, zero duplicates, zero unclassified rows, totals match; direct production/runtime imports: 11; direct test imports: 46; optional provider imports: 7; compatibility-only imports: 2; legacy optional imports: 2; direct tooling imports: 1; core contract leaks: 0; core implementation dependencies: 0; tooling dependency rows: 1.
- **Boundary:** PASS - 4,435 production files scanned; 10 allowed-zone imports; 5 grandfathered guarded imports; zero new forbidden imports.
- **LangGraph guard:** four findings, all pre-existing generated `applications/*/docker/runtime-context` findings; zero new canonical findings. Classified `EXPECTED_EXISTING_FINDINGS / NON_BLOCKING`.
- **Lock:** `uv lock --check` PASS.
- **CI presence:** `.github/workflows/unit-tests.yml` contains active `langchain-free-core` and `langchain-compatibility` jobs.

## Regression evidence

- Focused native/core/contract/runtime regression subset: **248 passed, 2 deselected**, one existing Pydantic warning.
- The initial 250-test selection also exposed two unrelated existing test issues: a stale reference to `scripts/check_langgraph_not_required.py` and missing optional `docling_core`; neither was changed.
- Optional live Ollama proof: **PASS**, one test; no model was pulled.

## Verdict

**`FINAL_GATE_COMPATIBILITY_FAILURE` - NOT READY_FOR_REVIEW.**

The evidence supports that the Intergrax default/core installation does not require LangChain or LangGraph, native contracts/providers are canonical, Native Ollama is the default, selected compatibility extras remain available in principle, and native RAG contracts are independent from LangChain `Document`. It does not claim that zero LangChain/LangGraph code exists.

`LCI-8A` was not started. Next: resolve or re-qualify the failing compatibility installation families, then rerun this final system gate.


---

## Chronological rerun - Transformers v4 constraint

**Task:** `LCI-FINAL-SYSTEM-GATE-RERUN-AFTER-TRANSFORMERS-V4-CONSTRAINT`
**Date:** 2026-08-10
**Platform:** Windows 10 (`win32`)
**Python:** 3.12.11
**Branch:** `development`
**Starting HEAD:** `c8efb2d13df82481f2ae6a6738dfb309b288e136`
**origin/development:** `c8efb2d13df82481f2ae6a6738dfb309b288e136`
**Required ancestors:** `c8efb2d1` and `079123ab` - present
**Later commits:** none at preflight

The previous qualification above remains preserved chronologically as
`FINAL_GATE_COMPATIBILITY_FAILURE`. Its root cause was the Transformers v5 /
Torch 2.2.2 incompatibility in `rag-langchain-loaders` and
`rag-langchain-splitters`. The accepted repair is commit
`c8efb2d13df82481f2ae6a6738dfb309b288e136`, with
`transformers>=4.41,<5` while retaining `torch==2.2.2` and
`sentence-transformers>=3.0`.

### Packaging and resolver fingerprint

- `[project].dependencies`: zero `langchain*`, zero `langgraph*`.
- Required core matrix: `torch==2.2.2`, `sentence-transformers>=3.0`,
  `transformers>=4.41,<5`.
- Compatibility extras present: `llm-langchain-ollama`,
  `rag-langchain-loaders`, `rag-langchain-embeddings`,
  `rag-langchain-splitters`, `langgraph-legacy`.
- Fresh 7B resolution: Torch `2.2.2+cpu`, sentence-transformers `5.7.0`,
  Transformers `4.57.6`, Tokenizers `0.22.2`.
- Fresh compatibility resolution: LangChain core `1.5.3`, community `0.4.2`,
  text-splitters `1.1.2`; Transformers remained `4.57.6`.

### Final gate results

| Gate | Result |
|---|---|
| 7B clean core | **PASS** - zero installed `langchain*`/`langgraph*`; core imports, registry, native Ollama/OpenAI, `KnowledgeDocument`, native RAG, and Nexus/Harness passed |
| `llm-langchain-ollama` | **PASS** |
| `rag-langchain-loaders` | **PASS** |
| `rag-langchain-embeddings` | **PASS** |
| `rag-langchain-splitters` | **PASS** |
| `langgraph-legacy` | **PASS** |
| Historical Torch probe | **PASS** - `is_torch_available=True`; `transformers.integrations.tensor_parallel` imported |
| Native LLM bounded regression | **PASS** - 281 passed, 27 deselected |
| `KnowledgeDocument` contract regression | **PASS** - 91 passed, 1 existing Pydantic warning |
| Live Ollama proof | **PASS** - 1 passed; no model pulled |

Native ownership remained unchanged: `LLMProvider.OLLAMA` resolves to
`NativeOllamaAdapter`, and the default RAG splitter remains native even when
compatibility packages are installed. Native OpenAI import/registry checks,
provider usage counters/fallbacks, capability/context behavior, and the
plain/tools/structured/stream paths passed.

### Audits and controls

- Inventory: **PASS** - 69 unique IDs, zero duplicate path/line/symbol, zero
  unclassified; 11 production/runtime imports, 46 direct test imports,
  7 optional provider imports, 2 compatibility-only imports, 2 legacy
  optional imports, 0 core contract leaks, 0 core implementation dependencies.
- Boundary: **PASS** - 4,440 production files scanned, 10 allowed-zone
  imports, 5 grandfathered guarded imports, 0 new forbidden imports.
- LangGraph guard: 4 known generated `applications/*/docker/runtime-context`
  findings, 0 new canonical findings; classified
  `EXPECTED_EXISTING_FINDINGS / NON_BLOCKING`.
- `uv lock --check`: **PASS**.
- CI workflow still contains active `langchain-free-core` and
  `langchain-compatibility` jobs.

An additional broad repository marker run was not used as the bounded gate:
it collected 3,569 passed and 61 unrelated failures across application,
scaffold, and concurrent RAG work. Those failures were separately classified
as existing/out-of-scope and were not repaired here.

## Final verdict

**`FINAL SYSTEM GATE - PASS / READY_FOR_REVIEW`**

This establishes that the default/core Intergrax installation requires
neither LangChain nor LangGraph, canonical/default LLM and RAG paths are
native, `KnowledgeDocument` is the native RAG contract, and selected
compatibility extras resolve reproducibly against the supported Torch 2.2.2 /
Transformers v4 matrix. It does not claim that Intergrax contains zero
LangChain or LangGraph code.

`LCI-8A` was not started. Next: `LCI-8A`.

During the rerun, later parallel commits `ee354401 fix(rag): serialize same-source replacement` and `0258cfe4 feat(vendor-knowledge): close Atlassian LKW readiness` advanced the branch. They were unrelated to this evidence-only gate and were preserved.
