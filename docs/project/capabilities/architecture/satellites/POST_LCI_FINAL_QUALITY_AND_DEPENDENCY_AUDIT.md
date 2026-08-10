# POST-LCI Final Quality, Stability and Dependency Audit

## Status and scope

- Task: `POST-LCI-FINAL-QUALITY-STABILITY-AND-DEPENDENCY-AUDIT`
- Starting HEAD: `d05b9e74fb6e25ceb62eb849d0aa42e857f665ac`
- Final audited HEAD: `36bce5b8ea325047079e689769cae5fb5e7b6b65`
- `origin/development`: `36bce5b8ea325047079e689769cae5fb5e7b6b65`
- Required ancestor `c44241e79a47174a0c1d6c95f8c1fbe6e606211f`: PASS
- Branch: `development`; platform: Windows 10 `10.0.26200`
- Host Python: 3.13.1; audit/test Python: 3.12.11; uv: 0.8.15
- Later commits observed: `9293308f`, `d2172ada`, `80155112`, `36bce5b8`.
- No production, RAG, packaging, test, script, or lockfile changes were made by this audit.
- RAG ownership fence was respected: RAG paths and RAG architecture documentation were read/tested only.

The unrelated staged path from the initial resume was preserved and was not touched. It cleared naturally before evidence staging. No reset, restore, unstage, stash, clean, rebase, checkout, amend, or force-push operation was used.

## Executive verdicts

| Area | Verdict | Evidence summary |
|---|---|---|
| Architecture | PASS_WITH_FINDINGS | Canonical native contracts/default chunking do not require LangChain; optional legacy provider paths still do. |
| Functional correctness | PASS_WITH_FINDINGS | KnowledgeDocument 91/91 passed; active indexing/RAG tests expose current contract drift. |
| Retrieval effectiveness | PASS_WITH_FINDINGS | Existing golden gate passed; direct native-vs-LangChain retrieval delta is inconclusive. |
| LLM parity | PASS_WITH_FINDINGS | 300 deterministic tests and current live Ollama proof passed. |
| RAG stability | INCONCLUSIVE | 229 passed, 12 failed, 2 skipped in current retrieval/indexing batch. |
| Memory/modality | PASS | 159 bounded Graph/Memory/Modality tests passed. |
| Performance | PASS_WITH_FINDINGS | Native adapter overhead is small; default Harness import and install footprint are heavy. |
| Dependency correctness | FAIL | `whisper` is the wrong runtime distribution; `botocore` has no direct import evidence. |
| Dependency modularity | FAIL | 45 core dependencies install provider, vector, media/UI, and local ML closures. |
| Dependency version safety | PASS_WITH_FINDINGS | Several high-impact packages have unbounded major ranges. |

## Executed checks and tests

- `git fetch --no-tags origin development`, branch/SHA/ancestor/status/staged-path checks: PASS.
- `uv lock --check`: PASS; resolver reported 385 packages.
- `uv pip check`: PASS; 254 installed packages compatible.
- Clean temporary environment outside repository: default install PASS; 204 distributions; 2,145,803,612 site-packages bytes; `import intergrax` 0.000235 s; `import intergrax.runtime.nexus` 0.000708 s; `import intergrax.harness` 54.207791 s.
- KnowledgeDocument contracts + conformance gate: 91 passed, 0 failed, 1 warning, 1.56 s.
- Knowledge/chunking/embedding/boundary batch: 244 passed, 1 failed, 1 warning, 20.68 s.
- Indexing/vectorstore/retriever/retrieval/reranker/tool batch: 229 passed, 12 failed, 2 skipped, 20.54 s.
- Graph RAG, memory, modality, compatibility document tests: 159 passed, 0 failed, 11.46 s.
- Selected native Ollama/provider tests: 143 passed, 0 failed, 19.38 s.
- Full non-network `tests/unit/llm_adapters`: 300 passed, 8 network tests deselected, 20.45 s.
- Existing retrieval metrics/golden gate: 5 passed, 0 failed, 5.83 s.
- Current live Ollama proof: 1 passed, 0 failed, 22.06 s.\n- Latest concurrent RAG backend test changes: 11 passed, 4 Chroma-service skips, 22.00 s.

Known non-quality/environment blockers:

- One chunking-batch failure imports unavailable `docling_core`.
- The 12 retrieval-batch failures are current test-double/expectation drift: missing `list_source_record_ids`, explicit tenant requirement, and missing fake `count`.
- Two tests skip because `fastembed` is absent and Chroma Rust upsert crashes on Windows.
- One expected malformed-model Pydantic serializer warning was emitted.

## Native architecture boundary audit

| Stage | Input | Output | LC required | Scope | Identity | Provenance |
|---|---|---|---|---|---|---|
| Parser | source string / `BaseDocumentParser` | `ParsedDocumentFragment` sequence + trace | NO in native pipeline; YES in some optional legacy loaders | preserved in materialization | assigned in materialization | assigned in materialization |
| Document | parser fragment | `KnowledgeDocument` | NO | YES | YES | YES |
| Normalization/metadata | native document/fragment | native text/metadata | NO | YES | YES | YES |
| Chunking | `Sequence[KnowledgeDocument]` | derived native documents | NO | YES | root/parent lineage | YES |
| Embedding | native texts/documents | float32 matrix/`EmbeddingResult` | NO for native providers | document unchanged | YES | YES |
| Indexing | documents + embeddings + vector manager | persisted vector IDs | NO | vector scope enforced | vector ID is document ID | document carried |
| Vector store | `VectorStoreRecord` + scope | immutable `VectorStoreHit` | NO in native ABI | tenant/namespace/workspace | YES | document carried |
| Retrieval | query + native scope/filter | `RetrievalHit` | NO | immutable on hit | YES | retained |
| Reranking | immutable `RerankerCandidate` | immutable `RerankerResult` | NO | unchanged | unchanged | unchanged |

`KnowledgeDocument` is frozen, strict, schema-versioned, tenant-scoped, namespace/workspace-aware, lineage-aware, provenance-aware, rejects reserved metadata, freezes metadata deeply, rejects non-finite/duplicate JSON, serializes deterministically as UTF-8, and rejects invalid UTF-8. The complete targeted suite passed. The knowledge package boundary test found no LangChain/LangGraph imports.

Canonical default chunking selects `recursive`. Native vector/retriever/reranker contracts contain no LangChain import and do not leak LangChain `Document`. Explicit compatibility imports remain in the LangChain splitter, LangChain Ollama adapter/embedding path, `intergrax.compat.langchain`, and selected optional document parser providers. These are compatibility-surface findings, not canonical default imports.

## Chunking comparison

Deterministic corpus: prose, headings, paragraphs, bullets, punctuation, Unicode, long unbroken text, short input, and multi-chunk input. Both used `chunk_size=80`, `chunk_overlap=16`.

| Metric | Native recursive | LangChain compatibility |
|---|---:|---:|
| Total chunks | 39 | 25 |
| Min/mean/max size | 25 / 56.85 / 80 | 18 / 71.36 / 80 |
| Empty chunks | 0 | 0 |
| Coverage | 100% for every source | 3 sources showed loss; 2 sources below 100% |
| Text loss | 0 chars | 21 chars in fixture |
| Duplicate chars | 528 | 116 |
| Duplication beyond configured overlap | 0 | 0 under measurement rule |
| Identity/scope/provenance | preserved for every chunk | preserved by wrapper |

Native boundaries were more granular and retained paragraph/heading separator text. LangChain produced fewer/larger chunks and trimmed content at several measured boundaries. Native semantics are equal or better for this corpus; byte identity is not required.

Chunking benchmark over 25 repetitions of six documents:

- native: median 2,357.75 documents/s, median 0.002545 s, p95 0.002909 s;
- compatibility: median 2,733.98 documents/s, median 0.002195 s, p95 0.004567 s.

Native is approximately 14% slower at median on this small fixture, with lower measured p95. Existing evaluation tooling does not parameterize the golden retrieval harness by chunking strategy; a direct native-vs-LangChain retrieval delta is therefore `INCONCLUSIVE` without inventing a new framework.

## Retrieval, indexing, vector, reranking, Graph RAG

Golden retrieval cases all passed. Metrics by case:

- keyword/semantic: R@1/R@3/R@5 `1/1/1`, MRR `1`, nDCG@5 `1`;
- exact SKU: `1/1/1`, MRR `1`, nDCG@5 `1`;
- graph entity: `1/1/1`, MRR `1`, nDCG@5 `1`;
- multi-hop graph: `0.5/1/1`, MRR `1`, nDCG@5 `1`;
- post-delete: expected empty result;
- agentic refinement: `1/1/1`, MRR `1`, nDCG@5 `1`.

The native indexing pipeline receives `KnowledgeDocument`, embeds without mutation, creates `VectorStoreRecord`, and returns persisted IDs. Native vector records/hits validate finite float32 vectors, immutable document copies, vector IDs, score/rank, and tenant/namespace/workspace scope. `RetrievalHit` is the provider-neutral result; the outer legacy `RetrievalChunk` adapter is typed and does not require LangChain. Reranker candidates/results preserve original score/rank and immutable document identity/scope/provenance while adding rerank/fusion scores and final rank. Provider-independent vector/reranker tests passed within the 229-pass batch.

Current RAG stability remains inconclusive because the same batch had 12 failures after the current vectorstore contract changed. The defect was reported only; no RAG file was changed.

Graph RAG was audited read-only. Current retriever accepts native vector hits and returns `RetrievalHit`; graph channel fusion, tenant scope, identity, and graph provenance trace tests passed.

## Native embedding parity

Deterministic embedding tests passed for OpenAI-compatible, vLLM-compatible, llama.cpp-compatible, and Ollama paths. Covered ordering, batching, dimensions, dtype, validation, empty input, retry/error propagation, lazy dimension handling, and native document preservation. No canonical embedding path requires LangChain. The Ollama provider retains only an explicit compatibility import.

## LLM regression and live proof

The registry resolves `LLMProvider.OLLAMA` to `NativeOllamaAdapter`. Full non-network adapter tests passed 300/300; selected native/conformance tests passed 143/143. Plain, tool, structured, JSON schema, stream, partial stream, usage, context window, capability, error, timeout, and configuration contracts were exercised.

Live server: `http://127.0.0.1:11434`; model: `qwen2.5:7b`; no pull performed. Native plain was non-empty; native tools emitted one `get_weather` call with `{"city":"Warsaw"}`; native structured output parsed `{"city":"Warsaw","temperature_c":20}`; stream emitted 30 partial and one final event with exact concatenation. Raw prompt/eval counters were plain `41/3`, tools `159/21`, structured `41/20`, stream `35/31`. Native usage source was `sdk`, compatibility baseline was `estimate`. Missing-model, invalid-schema, invalid-format, connection-refused, and bounded-timeout proofs passed.

## Memory and modality

159 Graph RAG, memory-vector wiring, user/session memory, multimodal contract, legacy `rag_answers` boundary, native vector input, and compatibility document tests passed. No expensive multimedia model was run.

## Performance

Using a deterministic fake Ollama client:

- plain message mapping/response conversion: median 15.60 microseconds/op, p95 19.07, 64,103.59 ops/s;
- stream mapping/assembly: median 20.16 microseconds/op, p95 23.84, 49,605.63 ops/s;
- reconstructed plain and stream content: `one two three`.

The clean default install has 204 distributions and approximately 2.15 GB of site-packages. Top-level and Nexus imports are fast, but public Harness import is approximately 54.2 seconds.

## Direct core dependency audit

`pyproject.toml` contains 45 direct `[project].dependencies` entries. Direct imports below are exact AST/search evidence in `intergrax`; lazy imports are named where applicable.

| Package | Direct imports | Owner | Required default | Optional candidate | Existing extra | Dev/type-only | Version policy | Finding | Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| torch | none; ST closure | HF embedding | YES via current bootstrap | YES after redesign | none | NO | exact | heavy closure | local extra |
| fastapi | 52 files | API/Harness | YES | NO for API package | harness duplicate | NO | unbounded | direct owner | bound major |
| uvicorn | 3 files | server/CLI | YES for server | YES headless | harness duplicate | NO | unbounded | server-specific | server extra |
| starlette | 5 files | middleware/auth | YES | NO while shipped | harness duplicate | NO | unbounded | direct owner justified | bind with FastAPI |
| httpx | 28 files | HTTP integrations | YES | partial | none | NO | unbounded | broad owner | bound major |
| python-multipart | none; FastAPI/MCP closure | HTTP/MCP | NO proven direct use | YES | none | NO | unbounded | transitive/feature candidate | API/MCP extra |
| pydantic | 674 files | contracts/config | YES | NO | harness duplicate | NO | unbounded | foundational | bound major |
| cryptography | 4 files | security/evidence | YES security paths | partial | none | NO | unbounded | direct owner | bound major |
| python-dotenv | 1 file | CLI env | NO headless | YES | harness duplicate | NO | unbounded | CLI-only | CLI extra |
| PyYAML | 5 files | YAML config | YES YAML APIs | partial | harness duplicate | NO | unbounded | config owner | bound major |
| tqdm | 3 files | loaders/media | NO minimal core | YES | none | NO | unbounded | loader/media-specific | media extra |
| openai | 11 files | LLM/embedding/vector | YES current Harness import | YES architecturally | llm-openai/groq/vllm/compat | NO | unbounded | provider in core | provider extra |
| anthropic | 1 file | Claude | NO | YES | llm-anthropic | NO | unbounded | provider in core | provider extra |
| mistralai | 1 file | Mistral | NO | YES | llm-mistral | NO | unbounded | provider in core | provider extra |
| ollama | 3 files | Ollama/media | NO | YES | llm-ollama | NO | unbounded | provider in core | provider extra |
| tiktoken | 2 files | token accounting | partial | YES | most llm extras | NO | unbounded | optional tokenizer | tokenizer extras |
| boto3 | 8 files | AWS/Bedrock | NO default | YES | llm-bedrock | NO | unbounded | provider-specific | AWS extra |
| botocore | none | boto3 closure | NO | YES | llm-bedrock | NO | unbounded | transitive duplicate | remove direct ownership |
| mypy-boto3-bedrock-runtime | 1 unconditional runtime file | Bedrock | NO default; YES adapter load | YES | llm-bedrock | NO current code | unbounded | provider runtime bloat | Bedrock extra/protocol |
| google-genai | 6 files | Gemini/GCP | NO | YES | llm-gemini/vertex | NO | unbounded | provider in core | provider extra |
| cohere | 2 files | Cohere/reranker | NO | YES | llm-cohere-native | NO | unbounded | provider in core | provider extra |
| chromadb | 2 files/lazy | Chroma | NO in-memory | YES | none | NO | exact | vector client in core | vector extra |
| qdrant-client | lazy opener | Qdrant | NO in-memory | YES | none | NO | unbounded | vector client in core | vector extra |
| pinecone | 2 files/lazy | Pinecone | NO in-memory | YES | none | NO | unbounded | SaaS vector in core | vector extra |
| sentence-transformers | 2 files + default import | local embeddings/rerank | YES current Harness import | YES redesign | none | NO | unbounded | heavy ML | `rag-local-embeddings` |
| transformers | tokenizer + ST closure | HF | YES current closure | YES | none | NO | `<5` | local ML | local extra |
| numpy | 27 files | RAG/vector/math | YES | NO | none | NO | exact | foundational | retain |
| pandas | 2 files | spreadsheet | NO minimal | YES | none | NO | `<3` | office-specific | office extra |
| beautifulsoup4 | 1 file | web extraction | NO | YES | none | NO | unbounded | web-specific | web extra |
| trafilatura | 1 file | web extraction | NO | YES | none | NO | unbounded | web-specific | web extra |
| python-docx | 1 file | DOCX | NO | YES | none | NO | unbounded | office-specific | office extra |
| openpyxl | lazy through pandas/declared parser | XLSX | NO | YES | none | NO | unbounded | office-specific | office extra |
| xlrd | none; pandas XLS support | legacy XLS | NO | YES | none | NO | unbounded | indirect format-specific | office extra |
| pytesseract | 3 files | OCR/vision | NO | YES | none | NO | unbounded | media-specific | vision extra |
| pillow | 2 files | image/OCR/PDF | NO | YES | none | NO | unbounded | media-specific | vision extra |
| PyMuPDF | 1 file | PDF/OCR | NO | YES | none | NO | unbounded | PDF-specific; optional loader LC-backed | PDF extra/migrate boundary |
| requests-cache | none found | web cache | NO | YES | none | NO | unbounded | unused candidate | remove from core later |
| whisper | 1 lazy file | audio transcription | NO | YES | none | NO | unbounded | WRONG_RUNTIME_DISTRIBUTION | separate media fix |
| yt-dlp | 1 file | YouTube/audio | NO | YES | none | NO | unbounded | media-specific | audio extra |
| webvtt-py | 2 files | VTT/media | NO | YES | none | NO | unbounded | media-specific | media extra |
| opencv-python-headless | 3 files | video/vision | NO | YES | none | NO | exact | media-specific | video extra |
| chardet | 1 file | text parser | partial | YES | none | NO | `<6` | parser-specific | text extra |
| streamlit | none found | UI | NO | YES | none | NO | unbounded | unused/UI bloat | UI extra |
| fastmcp | 4 optional files | MCP server | NO if disabled | YES | none | NO | unbounded | optional MCP in core | MCP extra/bound |
| mcp | none found; FastMCP transitive | MCP transport | NO | YES | none | NO | unbounded | direct transitive duplicate | remove or bound `<2` |

## Mandatory dependency findings

### mypy-boto3-bedrock-runtime

Only import: unconditional `from mypy_boto3_bedrock_runtime import BedrockRuntimeClient` in the Bedrock adapter. It is not `TYPE_CHECKING`-only. Classification: `PROVIDER_RUNTIME_BLOAT`, not `DEV_ONLY_RUNTIME_BLOAT`; move with Bedrock or replace with a protocol in a future change.

### whisper

Owning path: `intergrax/integrations/providers/document_parser/whisper/opens.py`. Expected API: `whisper.load_model()` and `model.transcribe()`. Resolved distribution: `whisper==1.1.10`, summary `Fixed size round-robin style database`, module `whisper.py`. Import fails on Windows while loading `libc`, and this is not the intended OpenAI speech-recognition distribution. Classification: `WRONG_RUNTIME_DISTRIBUTION`, P1.

### mcp/fastmcp

Project environment: `mcp==1.27.2`, `fastmcp==3.3.1`; clean default resolution: `mcp==1.29.0`, `fastmcp==3.4.6`. Intergrax has direct FastMCP ownership only in optional MCP composition modules and no direct `mcp` import. Direct mcp is therefore not justified by current source evidence; FastMCP already depends on it. If mcp remains public ABI, protect v1 with `<2` pending a v2 audit. FastMCP also needs an explicit tested major policy.

### boto3/botocore

Boto3 is directly imported by eight AWS/Bedrock/integration paths and belongs to an AWS/Bedrock extra. Botocore has no direct Intergrax imports and is already a boto3 transitive dependency: candidate duplicate.

### fastapi/starlette

FastAPI has 52 direct imports and Starlette five direct middleware/auth imports. Both have independent ownership; Starlette is not redundant merely because FastAPI depends on it. Bind their major versions together.

### provider SDK bundle

Provider registry resolution is lazy, but current default RAG/Harness imports default embedding modules eagerly. All provider SDKs remain core, so current `llm-*` groups are mostly `DESCRIPTIVE_DUPLICATION`, not `EFFECTIVE_OPTIONALIZATION`.

### local ML stack

`default_embedding_engine.py` imports `HFEmbeddingProvider` and other provider modules at module load; `HFEmbeddingProvider` imports `sentence_transformers`. Clean import evidence proves Nexus is fast but public Harness is 54.2 s. A future `rag-local-embeddings` extra is strongly indicated after lazy/default-provider redesign.

### vector clients, parsing/media, UI/MCP

Chroma, Qdrant, and Pinecone are provider-bound/lazy and unnecessary for minimal in-memory operation. Parsing/media dependencies are format- or media-specific. `requests-cache`, `streamlit`, and direct `mcp` have no direct Intergrax imports; FastMCP is optional MCP composition. These are optionalization candidates.

## Optional extras consistency

- Effective optional compatibility: `rag-langchain-loaders`, `rag-langchain-embeddings`, `rag-langchain-splitters`, `llm-langchain-ollama`, `langgraph-legacy`.
- Descriptive duplication: `harness-author`, `llm-openai`, `llm-anthropic`, `llm-mistral`, `llm-ollama`, `llm-gemini`, `llm-bedrock`, `llm-cohere-native`, `llm-all` while providers remain core.
- Partial/descriptive: `llm-groq`, `llm-vllm`, `llm-compat`, `llm-vertex`; their bases remain core.
- `dev`/`dev-ci` duplicate test/quality groups but also contain true development integrations.
- `integrations-*` are mostly truly optional.
- No extra was changed.

## Severity-ranked findings

### P0

- None.

### P1

- `F-01 WRONG_RUNTIME_DISTRIBUTION`: `whisper>=1.1` is Graphite Whisper, not the intended OpenAI speech runtime.
- `F-02 RAG_CONTRACT_DRIFT_UNRESOLVED`: 12 current indexing/vectorstore/RAG tests fail against post-closeout contract expectations; fenced and not repaired.
- `F-03 DEFAULT_HARNESS_HEAVY_IMPORT`: clean Harness import takes approximately 54.2 s due eager local HF embedding import.
- `F-04 CORE_PROVIDER_BUNDLE`: all provider SDKs, vector clients, media/UI packages, and local ML remain in the default runtime.

### P2

- `F-05 MYPY_BEDROCK_RUNTIME_BLOAT`: Bedrock type package is unconditional runtime import in provider-specific code.
- `F-06 BOTocore_DIRECT_DUPLICATE`: no direct import; boto3 already owns the transitive runtime.
- `F-07 EXTRAS_DESCRIPTIVE_DUPLICATION`: most LLM and Harness extras repeat core dependencies.
- `F-08 UNBOUNDED_MAJOR_RISK`: MCP/FastMCP, provider SDKs, FastAPI, Pydantic, and vector clients are not major-bounded.
- `F-09 OPTIONAL_PARSER_LANGCHAIN_BOUNDARIES`: selected optional parsers still require LangChain loaders.
- `F-10 DEFAULT_INSTALL_FOOTPRINT`: 204 distributions and approximately 2.15 GB clean site-packages.

### P3

- `F-11 CHUNKING_MEDIAN_COST`: native median was approximately 14% slower on the small fixture.
- `F-12 UNUSED_DEFAULT_CANDIDATES`: no direct imports found for requests-cache, streamlit, or direct mcp.
- `F-13 OPTIONAL_TEST_ENVIRONMENT`: Docling test requires absent `docling_core`.

## Recommended remediation roadmap (not executed)

1. Resolve the Whisper distribution/API mismatch with a dedicated Windows proof.
2. Close current RAG vectorstore test/contract drift under the separate RAG owner.
3. Make default embedding/provider registration lazy and remote-safe for headless Nexus/Harness.
4. Split local models into `rag-local-embeddings`.
5. Make `llm-*` extras effective and remove duplicated core provider ownership only with import gates.
6. Split vector clients and parsing/media into individual extras.
7. Remove or justify direct botocore, mcp, requests-cache, streamlit, and other unused declarations.
8. Add tested upper-major policies for MCP/FastMCP, provider SDKs, vector clients, FastAPI, and Pydantic.
9. Add a canonical retrieval mode that compares native and compatibility chunking using the existing metric tooling.

## Final recommendation

The LangChain migration is quality-safe for the canonical native document, chunking, embedding, indexing ABI, retrieval contract, reranking contract, Graph RAG, memory/modality, and native Ollama paths covered by executable evidence. It is not sufficient to declare the complete package dependency surface safe: wrong Whisper distribution, active RAG test drift, eager local ML import, all-provider core bundle, duplicated extras, and unbounded high-risk versions remain open.

Dependency slimming should start after RAG contract drift is closed, beginning with the Whisper distribution and default HF/local-ML boundary. The next coherent remediation block is: **provider/lifecycle ownership split with lazy default embedding bootstrap, `rag-local-embeddings`, effective `llm-*` extras, and a focused version-policy gate**, followed by a rerun of this audit.

