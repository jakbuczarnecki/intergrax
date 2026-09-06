# Embedding Provider Decision

**Task:** VPI-IMPLEMENTATION-4  
**Audit revision:** `development` @ `b9de0278411d97e3f935528f2e19da02be335644`  
**Scope:** embedding configuration and reference model selection only — no ingest, Qdrant bootstrap, or vector generation over WDC.

---

## Decision

**Canonical reference embedding configuration (reproducible default, swappable via env):**

| Setting | Value |
| --- | --- |
| Provider | `hf` |
| Model | `BAAI/bge-m3` |
| Dimension | `1024` |

Environment surface (scenario-owned, not global `INTERGRAX_EMBEDDING_*`):

```dotenv
VPI_EMBEDDING_PROVIDER=hf
VPI_EMBEDDING_MODEL=BAAI/bge-m3
VPI_EMBEDDING_DIMENSION=1024
```

Typed loader: `load_vpi_embedding_configuration()` in `application/config/embedding_configuration.py`.

**Reference model ≠ hard-coded model.** VPI business/domain code must not import vendors; operators may substitute any registered Intergrax embedding provider/model pair that satisfies the `EmbeddingProvider` contract and dimension validation.

---

## Why embedding configuration is swappable

- VPI reuses the platform `EmbeddingProvider` / `EmbeddingPipeline` ABI with Integrations-backed binding.
- Scenario configuration uses `VPI_EMBEDDING_*` prefix (parallel to `embedding_profile_from_env(prefix=...)` semantics for provider/model validation).
- Precedence follows canonical proof infrastructure: **process environment > scenario `.env` > scenario reference defaults** (`scripts/proof/intergrax_proof_environment.py`).
- Semantic derivation (`SemanticSearchRepresentation`) stays independent of embedding vendor; only the derived text is embedded downstream.

---

## Existing Intergrax embedding ABI

Audited on current `development`:

| Component | Role |
| --- | --- |
| `EmbeddingProvider` | `provider_name()`, `dimension()`, `embed(texts)` |
| `IntegrationProfile.embedding_provider` + `bind_embedding_provider()` | Canonical provider selection and runtime construction |
| `EmbeddingProfile` / `embedding_profile_from_env()` | Typed provider + model from env (default prefix `INTERGRAX_EMBEDDING`) |
| `create_default_embedding_pipeline()` | Bootstrap wiring via Integrations runtime binding |

**Registered providers (verified in `default_embedding_engine.py` + `EmbeddingProfile` validator):**

| Slug | Implementation module |
| --- | --- |
| `hf` | `HFEmbeddingProvider` (`sentence-transformers`) |
| `openai` | `OpenAIEmbeddingProvider` |
| `ollama` | `OllamaEmbeddingProvider` |
| `vllm` | `VllmEmbeddingProvider` |
| `llama_cpp` | `LlamaCppEmbeddingProvider` |

No `VPIEmbeddingProvider` was added — platform ABI is sufficient.

---

## Candidate models

| Candidate | Params | Typical dim | Notes |
| --- | --- | --- | --- |
| **BAAI/bge-m3** (selected reference) | ~568M | 1024 | Multilingual (100+), 8192 context, strong dense retrieval, no query/doc instruction split required for basic `embed()` ABI |
| Qwen/Qwen3-Embedding-4B | 4B | 2560 (base) | Instruction-aware; arena challenger |
| Qwen/Qwen3-Embedding-8B | 8B | 4096 (base) | Instruction-aware; arena challenger |
| Qwen/Qwen3-Embedding-0.6B (optional) | 0.6B | 1024 | Low-resource challenger |

Hosted alternatives (already supported when configured): OpenAI embedding models via `openai` provider; local served models via `ollama` / `vllm` / `llama_cpp`.

---

## Canonical default rationale

`BAAI/bge-m3` is the **reproducible reference configuration**, not a claimed VPI retrieval winner.

Why it fits VPI reference path:

- Open/local via `hf` provider
- Multilingual catalog content (WDC)
- 1024-d vectors — lower storage than 2560/4096 Qwen3 bases
- Works with simple `EmbeddingProvider.embed(texts)` — no instruction-aware query/document split in current contract
- Public weights — credible operator reproduction

**BGE-M3 sparse:** model can emit sparse representations, but VPI lexical retrieval remains on the Qdrant sparse/BM25 path (`INTERGRAX_RAG_QDRANT_SPARSE`). Dense reference uses BGE-M3 dense vectors only; sparse capability is a future arena challenger.

---

## Dimension / storage implications

Full corpus: **3,770,377** offers (selected WDC). Raw float32 vector footprint (vectors only):

| Dimension | Approx. raw size |
| --- | --- |
| 1024 | ≈ 15.4 GB |
| 2560 | ≈ 38.6 GB |
| 4096 | ≈ 61.8 GB |

Excludes Qdrant indexing, payload, metadata, and operational overhead.

`VPI_EMBEDDING_DIMENSION` expresses **expected** index compatibility. At bootstrap/ingest, `provider.dimension()` must match — **fail closed** on mismatch; never truncate or pad silently. Re-embedding + Qdrant collection rebuild required when model/dimension changes.

---

## Runtime configuration

| Variable | Purpose |
| --- | --- |
| `VPI_EMBEDDING_PROVIDER` | Registered slug (`hf`, `openai`, `ollama`, `vllm`, `llama_cpp`) |
| `VPI_EMBEDDING_MODEL` | Provider-specific model id |
| `VPI_EMBEDDING_DIMENSION` | Expected vector dimension for index compatibility |

Provider-native credentials (e.g. OpenAI API key) continue to use existing platform env vars when that provider is selected.

Example vendor swap (no VPI code changes):

```dotenv
VPI_EMBEDDING_PROVIDER=ollama
VPI_EMBEDDING_MODEL=nomic-embed-text
VPI_EMBEDDING_DIMENSION=768
```

---

## Qdrant compatibility

Embedding provider and Qdrant are independent layers:

```text
SemanticSearchRepresentation → EmbeddingProvider → vector → Qdrant adapter
```

Changing embedding vendor/model may require **Qdrant collection rebuild** because vector space and dimension change. It must **not** require changes to semantic derivation, `ProductCandidate`, verification, or identity logic.

---

## Alternatives

| Configuration | When |
| --- | --- |
| `hf` + `BAAI/bge-m3` @ 1024 | Canonical public reproduction |
| `openai` + supported embedding model | Hosted / API deployments |
| `ollama` / `vllm` / `llama_cpp` + compatible served model | Local inference stacks |
| Qwen3 4B/8B via `hf` | Arena evaluation only until instruction-aware contract is addressed |

Not every arbitrary model pair is guaranteed — provider must implement `EmbeddingProvider` and resolved dimension must match configuration.

---

## Known gaps

| Gap | Notes |
| --- | --- |
| Qwen3 instruction-aware behavior | Current contract is `embed(texts)` only — no `embed_query` / `embed_document` or task prompts. Exploiting Qwen3 instructions needs a future provider capability extension or scenario input formatting; **no Qwen-specific prompts in VPI domain**. |
| Bootstrap dimension gate | Contract defined (`validate_resolved_provider_dimension`); enforced at ingest/bootstrap in next task. |
| Full-corpus embedding generation | Deferred |
| Embedding arena / VPI retrieval metrics | Deferred — see below |

---

## Arena / future evaluation

**Canonical default ≠ benchmark winner.**

Later **embedding arena** must compare at minimum:

- BAAI/bge-m3
- Qwen/Qwen3-Embedding-4B
- Qwen/Qwen3-Embedding-8B

against **VPI-specific retrieval metrics** on WDC-derived representations. Do not substitute general MTEB leaderboard scores for VPI accuracy claims.

---

## BGE-M3 live compatibility probe

**Date:** 2026-09-02  
**Method:** `HFEmbeddingProvider(model_name="BAAI/bge-m3")` with `uv run --extra rag-local-embeddings`.

| Check | Result |
| --- | --- |
| Provider import | OK (`sentence-transformers` available with extra) |
| Model load + `dimension() == 1024` | **UNVERIFIED** |
| Multilingual `embed()` shape/finite values | **UNVERIFIED** |

**Blocker:** `torch.load` security gate — environment `torch` below 2.6; `SentenceTransformer` load aborted with CVE-2025-32434 mitigation (`ValueError`). Probe did not fake PASS.

**Operator unblock:** upgrade `torch` to ≥ 2.6 (or load weights via safetensors-only path when supported) and re-run bounded manual probe. Architecture selection of BGE-M3 + `hf` remains valid pending environment verification.

---

## Future bootstrap manifest fields

Scenario-owned `VpiIndexEmbeddingIdentity` (see `application/config/embedding_configuration.py`):

| Field | Purpose |
| --- | --- |
| `embedding_provider` | Registered slug used at ingest |
| `embedding_model` | Model id |
| `embedding_dimension` | Qdrant dense collection compatibility |
| `search_representation_derivation_version` | e.g. `v2` |
| `dataset_checksum` | WDC artifact integrity |
| `embedding_configuration_version` | Scenario embedding config revision (`v1`) |

Mismatch between manifest/index dimension and active configuration → fail closed; rebuild required.

---

## Next task

**VPI-IMPLEMENTATION-5: Reusable Storage Bootstrap & Ingest**

- PostgreSQL schema + Qdrant collection bootstrap
- Wire `DerivedOfferSearchRepresentation` ingest
- Enforce embedding dimension validation at bootstrap
- Validation gate (counts, checksum, READY)

Do **not** start: full 3.77M embedding batch, fusion, verification, proof evaluator.
