# LCI-7C â€” LangChain compatibility installation gate

## Verdict

**READY_FOR_REVIEW**

Validated checkout SHA: `a61f091e3e2717990a0ee2a24961d5f9d8ed023e`
Platform: Windows 10 (`win32`)
Python: 3.12.11
Validated: 2026-08-09

The reported loader/splitter Torch/Transformers defect is not reproducible in
fresh isolated installations from this checkout. Both target families and all
three control families pass. No dependency constraint was added because the
installed versions satisfy their upstream metadata and no failing import or
construction remains to constrain.

The earlier defect evidence is therefore classified as a stale or
non-reproducible observation, not as a confirmed packaging incompatibility.
No production or RAG files were changed, and LCI-7D was not started.

## Forensic compatibility record

### Root cause classification

- `torch`: `2.2.2+cpu` (declared as `torch==2.2.2`)
- `sentence-transformers`: `5.2.0`
- `transformers`: `4.57.6`
- `tokenizers`: `0.22.2`
- `langchain-text-splitters`: `1.1.0`
- `langchain-community`: `0.4.1` in the loaders environment
- `langchain-core`: `1.2.7`
- first failing import: none; `torch`, `transformers`,
  `sentence_transformers`, and `langchain_text_splitters` all imported
- classification: **E â€” different actual root cause / prior failure
  non-reproducible**

The prior defect report named `NameError: name 'torch' is not defined`, but did
not preserve an independently verifiable resolved-version set. With the exact
versions above, both `uv pip check` runs pass and the target imports pass.

### Compatibility ownership decision

- `sentence-transformers 5.2.0` declares `torch>=1.11.0` and
  `transformers>=4.41.0,<6.0.0`.
- `transformers 4.57.6` declares `tokenizers>=0.22.0,<=0.23.0`; its Torch
  requirement is `torch>=2.2` for the relevant optional Torch surfaces.
- `torch 2.2.2+cpu` and `tokenizers 0.22.2` satisfy those constraints.
- Torch was not changed.
- No `sentence-transformers` or `transformers` constraint was added.
- The absence of a new constraint is intentional: a bounded pin would not be
  supported by the observed failure or package metadata.

## Controls

- 7B clean-core installation: **PASS**
- Installed package origin: temporary environment `site-packages`, outside the checkout
- `PYTHONNOUSERSITE=1`: **PASS**
- Core `[project].dependencies`: zero LangChain/LangGraph distributions
- Missing-extra control (Ollama and representative RAG providers): **PASS**
- Each extra was installed in its own temporary Python 3.12 environment; no shared `.venv` was used.

## Compatibility families

| Extra | Direct distributions | Installed compatibility closure | Result |
|---|---|---|---|
| `llm-langchain-ollama` | `langchain-core`, `langchain-ollama` | `langchain-core`, `langchain-ollama` | PASS |
| `rag-langchain-loaders` | `langchain-community` | `langchain-classic`, `langchain-community`, `langchain-core`, `langchain-text-splitters` | PASS |
| `rag-langchain-embeddings` | `langchain-ollama` | `langchain-core`, `langchain-ollama` | PASS |
| `rag-langchain-splitters` | `langchain-text-splitters` | `langchain-core`, `langchain-text-splitters` | PASS |
| `langgraph-legacy` | `langgraph` | `langchain-core`, `langgraph`, `langgraph-checkpoint`, `langgraph-prebuilt`, `langgraph-sdk` | PASS |

The loader family passed `langchain-community` import and provider loader
construction. The splitter family passed
`LangChainRecursiveChunkingStrategy` import/construction/chunking and confirmed
that the native splitter remains the default and the LangChain splitter is not
implicitly registered.

The Ollama family passed deterministic plain, tools, structured, and stream
checks; `LLMAdapterRegistry.create(OLLAMA)` returned `NativeOllamaAdapter`.
Embeddings passed the deterministic mocked `OllamaEmbeddings` ABI. LangGraph
passed distribution and legacy module boundary checks.

## Versions before / after

No packaging change was made, so the resolver result before and after the
review is identical in both target environments:

- `torch==2.2.2+cpu`
- `sentence-transformers==5.2.0`
- `transformers==4.57.6`
- `tokenizers==0.22.2`
- `langchain-text-splitters==1.1.0`
- loaders: `langchain-community==0.4.1`, `langchain-core==1.2.7`

## Tests and audits

- Clean-core 7B gate: **PASS**
- All five 7C compatibility families: **PASS**
- Targeted compatibility tests: **20 passed** (Ollama 5, splitter 4, loaders 11)
- Targeted loader/splitter import and construction smoke: **PASS**
- `uv pip check` in both target environments: **PASS**
- `validate_langchain_inventory.py`: **PASS**
- `check_langchain_boundary.py`: **PASS**
- `uv lock --check`: **PASS**
- `git diff --check`: **PASS** (line-ending warning only)

## Changed files

- `docs/project/capabilities/architecture/satellites/LANGCHAIN_COMPATIBILITY_INSTALLATION_GATE.md`

`pyproject.toml` and `uv.lock` are unchanged; there is no unrelated lockfile
drift. No RAG implementation or RAG test path was modified.

LCI-7A: **APPROVED**
LCI-7B: **APPROVED**
LCI-7C: **READY_FOR_REVIEW**
LCI-7D: **NOT STARTED**
## Stability re-opened â€” 2026-08-09

The subsequent final system gate reproduced the loader and splitter failure in fresh isolated environments. The earlier classification as stale or non-reproducible is superseded. LCI-7C stability qualification is reopened under `LCI-7C-STABILITY-1-REPRODUCIBLE-TORCH-IMPORT-FAILURE-FORENSICS`.

The exact upstream traceback is `transformers/integrations/tensor_parallel.py:465`, where `_AllReduceBackward(torch.autograd.Function)` is evaluated while `transformers 5.14.1` has disabled Torch availability for installed `torch 2.2.2+cpu`. The prior PASS used a different resolved tuple (`transformers 4.57.6`, `sentence-transformers 5.2.0`, `langchain-core 1.2.7`, and related compatibility versions). See `LANGCHAIN_COMPATIBILITY_STABILITY_FORENSICS.md` for the reproducibility matrix and package provenance.

No packaging or runtime change was made. A dependency constraint decision is required before requalification; LCI-8A remains not started.
