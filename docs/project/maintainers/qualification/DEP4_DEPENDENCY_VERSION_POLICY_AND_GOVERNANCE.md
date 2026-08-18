# DEP-4 — Dependency version policy and governance

Status: `READY_FOR_REVIEW`

Starting HEAD: `a467793fbe35775bfb890c834b708de42474f54e`
Validated parent after later parallel commit: `cec5355e`
Required ancestor: `cd398c47ac1ac917b80059b21238d19b488ae9fc`
Branch/origin: `development` / `origin/development` at validation start

## Scope

DEP-4 changes dependency declarations, resolver constraints, and governance
checks only. No provider behavior, RAG algorithm, retrieval/indexing contract,
tenant/scope semantics, chunking, embedding algorithm, or RAG documentation
was changed. The `pandas` direct declaration was removed from core after audit:
its only production import is the optional `openpyxl` spreadsheet adapter, so
it is now owned by `parsing-office`.

## Core direct dependency inventory

| Package | Owner | Core category | Direct import evidence | Version policy | Major-bound status | Transitive duplicate risk | Recommendation |
|---|---|---|---|---|---|---|---|
| `fastapi` | Harness/server | `CORE_SERVER` | `intergrax/harness/app.py:10` | `>=0.115,<1` | bounded | intentional shared with `harness-author` | retain |
| `uvicorn` | Harness/server | `CORE_SERVER` | lazy server launch in `intergrax/harness/app.py:195` | `>=0.30,<1` | bounded | intentional shared with `harness-author` | retain |
| `starlette` | FastAPI middleware contracts | `CORE_SERVER` | `intergrax/fastapi_core/middleware/*`; direct compatibility anchor | `>=0.41,<1` | bounded | FastAPI transitive, direct anchor deliberate | retain |
| `httpx` | Web fetch/runtime HTTP | `CORE_FOUNDATION` | `intergrax/websearch/fetcher/http_fetcher.py:8` | `>=0.27,<1` | bounded | no finding | retain |
| `python-multipart` | FastAPI form/server surface | `CORE_SERVER` | FastAPI multipart surface | `>=0.0.9,<1` | bounded | no finding | retain |
| `pydantic` | Contracts/configuration | `CORE_FOUNDATION` | `intergrax/tools/*`, runtime/vendor contracts | `>=2.7,<3` | bounded | no finding | retain |
| `cryptography` | Evidence/attestation | `CORE_FOUNDATION` | `intergrax/runtime/execution_evidence/*`, attestation | `>=42,<47` | bounded | no finding | retain |
| `python-dotenv` | CLI/environment bootstrap | `CORE_FOUNDATION` | `intergrax/cli/run.py:27` | `>=1,<2` | bounded | intentional shared with `harness-author` | retain |
| `PyYAML` | Prompt/policy serialization | `CORE_FOUNDATION` | `intergrax/runtime/replay/policy_loader.py:6` | `>=6,<7` | bounded | intentional shared with `harness-author` | retain |
| `tqdm` | Core ingestion progress | `CORE_FOUNDATION` | `intergrax/rag/document_loaders/documents_loader.py:12` | `>=4.66,<5` | bounded | no finding | retain |
| `openai` | Native embeddings/LLM surface | `CORE_FOUNDATION` | `intergrax/rag/embedding/providers/*`, OpenAI adapters | `>=1,<3` | bounded | intentional shared with LLM extras | retain |
| `tiktoken` | Canonical token accounting | `CORE_FOUNDATION` | `intergrax/llm_adapters/contracts/llm_adapter.py:15` | `>=0.7,<1` | bounded | intentional shared with LLM extras | retain |
| `boto3` | AWS integration/server surface | `CORE_SERVER` | AWS/S3/Bedrock integration openers | `>=1.34,<2` | bounded | intentional shared with `llm-bedrock` | retain |
| `numpy` | Native vector/RAG contracts | `CORE_FOUNDATION` | `intergrax/rag/*` vector/index contracts | `==1.26.4` | exact | no finding | retain |
| `chardet` | Canonical native text fallback | `CORE_FOUNDATION` | lazy fallback in `intergrax/rag/document_loaders/parsers/text_smart_parser.py:25` | `>=5.2,<6` | bounded | no finding | retain |

Core total after DEP-4: **15 direct dependencies**. No core entry is
classified solely as provider, vector, parser, media, UI, or integration
optional.

## Version-policy table

High-risk direct and major optional families are bounded as follows:

```text
EXACT_PIN       numpy==1.26.4; chromadb==1.4.1
BOUNDED_MAJOR   FastAPI/Starlette/Uvicorn/HTTPX <1
                Pydantic <3; cryptography <47
                OpenAI <3; boto3 <2; tiktoken <1
                Anthropic <1; Mistral <2; Ollama <1
                google-genai <2; Cohere <6
                qdrant-client <2; pinecone <9
                FastMCP <4; openai-whisper calendar bound <20250626
                LangChain community <0.5, Ollama/splitters <2
                LangGraph <2; pandas <3
```

FastAPI and Starlette retain direct declarations because the middleware code
uses Starlette directly; both are bounded below `1` to prevent incompatible
major-family resolution. FastMCP `3.3.1` is qualified with `<4`; `mcp` remains
transitive and has no direct declaration. Parser `langchain-community` remains
an `OPTIONAL_LEGACY_IMPLEMENTATION` in `parsing-office` and `parsing-pdf`.

## Governance gate

Checker: `scripts/maintenance/check_dependency_ownership.py`

The checker is deterministic and offline. It verifies:

- every core declaration is in the reviewed `CORE_ALLOWLIST`;
- forbidden optional capability families, LangChain, and LangGraph stay out of
  core;
- `llm-all` contains native SDKs only;
- direct `mcp` and known transitive-only declarations are rejected;
- accidental core/extra duplicate ownership is rejected while documented
  shared owners remain allowed;
- every occurrence of a designated high-risk package has an upper major bound,
  with exact-pin exceptions visible in source.

Result: `dependency governance: OK`; 15 core entries and 102 optional direct
declarations checked. CI wiring is in the lightweight governance smoke step of
`.github/workflows/unit-tests.yml`.

## Resolver qualification

Fresh Python 3.12.11 default install, no extras:

```text
PASS
distributions: 42
site-packages: 124.40 MiB
KnowledgeDocument: PASS
default embedding registry: hf (NativeOllamaAdapter default invariant)
optional capability modules loaded: 0
```

The DEP-3 reference was 45 distributions / 171.46 MiB. The reduction is
explained by removing the stale core `pandas` declaration and its core-only
resolution footprint; `pandas` resolves through `parsing-office`.

Frozen representative resolves, with no API calls or model downloads:

```text
PASS  llm-openai
PASS  llm-ollama
PASS  rag-local-embeddings
PASS  vector-qdrant
PASS  parsing-office
PASS  media-whisper
PASS  mcp
PASS  llm-all
PASS  llm-all + rag-local-embeddings
PASS  vector-qdrant + rag-local-embeddings
PASS  parsing-office + parsing-pdf
PASS  mcp + harness-author
```

## Tests and audits

```text
dependency governance + DEP-1/DEP-3 declarations and selected boundaries: 58 passed
clean-core smoke: passed
representative resolver matrix: 12/12 passed
broader provider regression selection: 1566 passed, 18 unrelated failures
```

The broader provider selection exposed 18 pre-existing
`test_provider_runtime_cutover.py` failures unrelated to dependency policy;
they were not modified. The focused dependency/boundary selection is green
(58 passed). The two declaration assertions affected by the new version
bounds were updated and pass.

Required audit commands:

```text
uv run python scripts/docs/validate_langchain_inventory.py  PASS
uv run python scripts/maintenance/check_langchain_boundary.py PASS
uv lock --check                                             PASS
git diff --check                                             PASS
```

## Exceptions and deferred items

- `parsing-office` and `parsing-pdf` retain `langchain-community` as an
  optional legacy implementation; replacing it is separate technical debt.
- Shared core/provider declarations are intentional and listed in the checker.
- No direct `mcp` constraint was added; FastMCP owns its transitive MCP family.
- Existing provider cutover test drift remains outside DEP-4.
- Known tenant-id/RAG test drift remains owned by the separate RAG workstream.

## Change procedure

Change the declaration, its reviewed owner/bound, and this qualification
evidence together. Run the checker, focused declaration and boundary tests,
fresh resolver matrix, `uv lock --check`, and `git diff --check`. The gate has
no baseline-update or global-ignore mode.
