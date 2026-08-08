# Modality

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/MODALITY.md`](../maintainers/plans/MODALITY.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 29  
**Audit instruction:** [`audit/MODALITY.md`](../maintainers/audit/MODALITY.md)  
**Last updated:** 2026-06-20 — Modality Production Boundary (plane-specific maturity; cross-layer disambiguation)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (MODALITY canon).

- **Implement / audit default:** modality adapters hub. Tool surface: [`satellites/MODALITY_tool_surface_detail.md`](satellites/MODALITY_tool_surface_detail.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/MODALITY.md`](../maintainers/plans/MODALITY.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/MODALITY.md`](../technical/guides/audit_slices/MODALITY.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/MODALITY_tool_surface_detail.md`](satellites/MODALITY_tool_surface_detail.md) | tool surface detail |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.
## Why this document exists

Harness AI at scale needs more than text LLMs: images, audio, dedicated CV detectors (YOLO, SAM, OCR pipelines), embeddings, rerankers, and batch classifiers. Intergrax already implements **parts** of this (Whisper ingest, HF embeddings, image smart loaders) without a single architectural name.

This file is the **modality index**: which plane owns what, how to extend it, and what agents may call.

---

## Modality Production Boundary

Modality in Intergrax is **not** a single monolithic capability. Support is split into **distinct planes** with separate owners, access paths, maturity claims, and production constraints.

**Normative rule:** Modality support is split into distinct planes. A component **MUST NOT** be treated as production-ready for all modalities only because one modality plane is implemented or documented.

Multimodal behavior **MUST NOT** be conflated with:

- Integration Library adapters (storage, speech SaaS, remote inference hosts),
- RAG ingest and knowledge indexing,
- ToolRuntime side effects and agent-invokable tools,
- LLM adapter routing and model profiles,
- Dedicated inference without deployment/resource profiles.

Agents and Tier-3 applications consume modality through **approved planes only** — not direct SDK calls, agent-local model code, or undifferentiated "modality is done" claims.

**Cross-refs:** [`SYSTEM_INVARIANTS.md`](../technical/guides/SYSTEM_INVARIANTS.md) · [`MATURITY_TAXONOMY.md`](../technical/guides/MATURITY_TAXONOMY.md) · [`LLM_ADAPTERS.md`](LLM_ADAPTERS.md) · [`INTEGRATIONS.md`](INTEGRATIONS.md) · [`RAG.md`](RAG.md) · [`TOOLS.md`](TOOLS.md) · [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine) · [`MEMORY.md`](MEMORY.md)

---

## Modality planes

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  Plane A — Generative cognition (multimodal LLM)                        │
│  intergrax/llm_adapters/  —  NOT in Integration Library (§44.10)        │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│  Plane B — Media → text (ingest / indexing)                             │
│  integrations/document_parser + rag/ingest + smart loaders              │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│  Plane C — Dedicated inference (CV, classical ML, served models)        │
│  model_inference/ (planned) + integration hosts (HF Endpoint, Triton…)  │
└─────────────────────────────────────────────────────────────────────────┘
         ▲
         │  production path for agents
         └──────── ToolRuntime.invoke(tool_id) only
```

| Plane | Question it answers | Agent access |
|-------|---------------------|--------------|
| **A** | “Reason over image/audio/text in dialog?” | Via Nexus → `LLMProfile` (native vision/audio APIs) |
| **B** | “Turn file/URL into indexable text?” | Via tools (`rag.ingest_document`, parsers) — not direct SDK |
| **C** | “Run YOLO / sklearn / ONNX on bytes?” | Via tools (`vision.detect`, `ml.predict`, `speech.synthesize`) |

---

### Plane A — Generative multimodal LLM

**Purpose:** Model calls that accept or produce multimodal content through an approved LLM adapter/profile.

**Owner:** LLM adapter layer + model routing/profile.

**Allowed:**

- image/audio/document inputs to multimodal models if supported by selected provider/model,
- multimodal output where model/profile supports it,
- model capability declarations,
- token/cost/context controls.

**Must not:**

- bypass `LLMAdapter`,
- be treated as OCR/RAG ingest,
- own media storage,
- own product workflow.

**Implementation (as-built):**

- **Module:** `intergrax/llm_adapters/` only.
- **Capabilities (target contract):** `supports_vision`, `supports_audio_input`, `supports_audio_output` on `LLMAdapter`.
- **Messages:** `intergrax/llm/messages.py` — `AttachmentRef` (`image`, `audio`, `video`, …); adapters MUST map attachments to vendor content parts when capability flags are true.
- **When to use:** interactive reasoning, captioning in chat, tool planning with visual context.

**Do not** register OpenAI/Gemini/Claude as `integration` slugs.

---

### Plane B — Media/document ingest and indexing

**Purpose:** Convert media/documents into normalized text, chunks, metadata, embeddings or retrieval artifacts.

**Owner:** Document/media ingestion services, RAG ingest, parser integrations, approved tools.

**Allowed:**

- OCR,
- document parsing,
- audio transcription,
- image metadata extraction,
- media-to-text normalization,
- chunking and indexing,
- provenance preservation.

**Must not:**

- write directly to agent memory,
- bypass RAG ingest or approved knowledge indexing,
- bypass provenance and traceability,
- be treated as a general CV reasoning engine.

**Implementation (as-built):**

Already shipped (see implementation plan M.6 / M-RAG):

| Slug / component | Role |
|------------------|------|
| `whisper`, `yt_dlp` | Audio → transcript (ingest) |
| `docling`, `pypdf`, … | Document parsers |
| `ImageSmartLoader` / handlers | OCR + optional LLM caption → text for index |
| `HFEmbeddingProvider` | Local SentenceTransformers (`INTERGRAX_DEFAULT_HF_EMBED_MODEL`) |
| `splade` sparse encoder | Optional hybrid sparse (`INTERGRAX_RAG_SPARSE_ENCODER`) |

**Rule:** ingest output is **text (or embeddings)** in the knowledge layer — not a substitute for Plane C detectors in safety-critical paths unless policy allows.

---

### Plane C — Dedicated inference / CV / classical ML

**Purpose:** Specialized inference outside generic LLM calls, such as CV models, classifiers, detectors, rerankers, embedding models, custom ML models.

**Owner:** Dedicated inference service / `model_inference` layer / approved integrations and tools.

**Allowed:**

- classifier inference,
- object detection,
- OCR models where configured as dedicated inference,
- rerankers,
- embedding generation,
- custom model hosting,
- GPU/remote inference hosts if configured.

**Must not:**

- run as hidden agent-local code,
- bypass ToolRuntime when agent-invokable,
- bypass deployment/resource profiles,
- silently use heavy local models in production,
- be described as production-ready without maturity/evidence.

For **production CV** and **classical ML** where a multimodal LLM is the wrong tool (latency, cost, determinism, regulated bounding boxes).

#### C.1 Vision inference engine (extensible)

**Target module:** `intergrax/model_inference/` (Tier-0, planned Phase W-ML).

**Contract:** `VisionInferenceAdapter` — uniform API over heterogeneous backends:

| Backend family | Examples | Typical use |
|----------------|----------|-------------|
| **Ultralytics / YOLO** | YOLOv8–v11, RT-DETR exports | Object detection, segmentation, pose |
| **ONNX Runtime** | Exported `.onnx` from PyTorch/TF | Edge, cross-vendor deploy |
| **OpenVINO** | Intel-optimized models | On-prem CPU/GPU |
| **TensorRT** | NVIDIA serving | Low-latency GPU |
| **TorchScript / `.pt`** | Local weights | Lab, air-gapped |
| **Remote serving** | Triton, TorchServe, HF Inference Endpoints, SageMaker, Replicate | Horizontal scale |

**Registry pattern (same as LLM/RAG):**

```text
VisionModelProfile  →  VisionInferenceRegistry  →  VisionInferenceAdapter
     │                        │
     │                        └── slug: yolo_ultralytics | onnxruntime | openvino | triton_grpc | …
     └── model_id, version, input_schema, output_schema, risk_tier, device_policy
```

**Structured output (required for tools):** `DetectionResult`, `SegmentationResult`, `OcrRegionResult` — JSON-schema friendly; trace stores model slug, version, latency_ms, input hash (not raw bytes in trace by default).

**Execution placement:**

- **In-process** — only for lightweight ONNX / small models with explicit memory/GPU quotas.
- **Worker pool** — `ModalityExecutionProfile` + `ThreadPoolModalityInferenceExecutor` offloads heavy slugs (`yolo_ultralytics`, `vision_serving`, `huggingface_inference`).
- **Celery** — `ModalityExecutionMode.CELERY` + `CeleryModalityInferenceExecutor` dispatch serialized jobs (`intergrax.modality.run_job`) when `INTERGRAX_MODALITY_CELERY_BROKER_URL` (or `CELERY_BROKER_URL`) is set; falls back to thread pool when the broker is missing or dispatch fails. `wire_modality_extras()` registers the task on the shared `message_bus` Celery bundle when a broker is configured (`modality_celery_wiring.py`).
- **Remote endpoint** — preferred at high scale; integration slug under `ml_inference_host` or `vision_serving`.

#### C.2 Classical ML (non-CV)

**Contract:** `ModelInferenceAdapter` — sklearn, XGBoost, ONNX classifiers, small torch models.

| Concern | Harness approach |
|---------|------------------|
| Artifact | `ModelArtifact` record: id, version, schema, owner, risk_tier, license |
| Invocation | Tool `ml.predict` / `ml.batch_predict` |
| Versioning | SemVer + immutable artifact URI (object storage) |
| Eval | Reuse V-EVAL + braintrust/phoenix/wandb observability slugs |

#### C.3 Hugging Face — four roles (do not conflate)

| Role | Layer | Example |
|------|-------|---------|
| Embeddings | `rag/embedding` | `HFEmbeddingProvider` |
| Sparse / rerank | `rag/` or integrations | SPLADE, `jina_rerank` |
| Hub artifacts | Governance | Pin revision, license scan, CVE policy (V-SEC) |
| Hosted inference | Integration / remote adapter | HF Inference Endpoints, TGI |

**Rule:** Hugging Face Hub ≠ Nexus hot path. Heavy weights load in workers or remote hosts.

---

## Integration categories (modality-related)

| Category | Contract | Shipped slugs | Extension |
|----------|----------|---------------|-----------|
| **speech_provider** | `SpeechProviderBackend` (TTS/STT SaaS) | `elevenlabs`, `deepgram` | Manifest + factory or `IntegrationPlugin` — **slug identity only** ([ADR-MOD-001](../technical/adr/entries/2026-06-19/ADR-MOD-001.md)) |
| **vision_serving** | Remote CV server gRPC/REST | `triton` | Same open-catalog rules |
| **ml_inference_host** | Managed model endpoint | `replicate`, `huggingface_inference` | Same open-catalog rules |

**Planned slugs (not yet registered):** `azure_speech`, `openai_tts`, `torchserve`, `roboflow`, `sagemaker`, `azure_ml`, `vertex_prediction`.

**Existing (non-modality-C):** `document_parser` (ingest), `rerank_provider`, observability (`wandb`, `arize`, `phoenix`) for **eval**, not training.

### Plane C — Speech (TTS/STT) — canonical wiring

Speech SaaS vendors are **Integration Library** providers, not a closed platform enum.

```text
IntegrationManifest (slug) + factory
    → SpeechProviderBackend (Protocol instance)
        → wire_integration_tool_context()
            → IntegrationSpeechAdapter (slug-labelled bridge)
                → speech.synthesize / speech.transcribe (ToolRuntime)
```

| Rule | Detail |
|------|--------|
| **Single path** | Tier-3 hosts resolve `IntegrationProfile.speech_provider` (manifest, plugin class, slug `str`, or pre-built instance) — same binding model as other integration slots. |
| **No enum** | Do **not** use `SpeechProvider` enum or enum-coerced profiles — removed per MOD-SPEECH-ARCH (hard cutover, no deprecation phase). |
| **Env defaults** | `INTERGRAX_SPEECH_PROVIDER=<slug>` resolves against the registered integration catalog, not a platform enum. |
| **Plane B ingest** | File/audio transcription for RAG remains `document_parser/whisper` — not a substitute for `speech.transcribe` in dialog. |
| **Extension** | Third-party packages: `IntegrationPlugin` with category `speech_provider`; see [`INTEGRATIONS.md`](INTEGRATIONS.md) §Open catalog. |

**Legacy removed (MOD-SPEECH-ARCH):** `SpeechProvider` enum, `speech_provider_for_slug()` hardcoded mapping, enum-only `SpeechProfile` coercion, parallel enum backend in `wire_modality_extras()` when integration slot is configured.

---

## Modality responsibility boundary

| Concern | Owner |
|---|---|
| Multimodal model call | LLMAdapter / model profile |
| Model capability declaration | Model catalog / LLM profile |
| Document/media parsing | Parser integration / ingest service |
| RAG indexing | RAG ingest / knowledge service |
| Memory write | Memory service / policy |
| Agent decision using media-derived context | Tier-2 agent |
| Agent-invokable media processing | ToolRuntime + approved tool |
| Dedicated CV/ML inference | Model inference service / approved integration |
| Media artifact storage | Storage integration / application profile |
| Provenance and traceability | RuntimeEvent / observability spine |
| Product workflow | Tier-3 application + agents |

---

## Disallowed modality patterns

Intergrax components **MUST NOT**:

- treat all modality support as one layer with one maturity level,
- call provider multimodal APIs directly from agents,
- call OCR/CV libraries directly from agents in production,
- store media-derived facts directly into long-term memory without policy and provenance,
- mix RAG knowledge indexes with user/session memory indexes,
- bypass ContextCompiler when adding media-derived context to LLM calls,
- bypass ToolRuntime for agent-invokable media side effects,
- run heavy local inference models in production without deployment/resource profile,
- treat image/audio/document parsing as proof of semantic understanding,
- describe modality as production-ready without plane-specific maturity/evidence,
- use media artifacts without retention/privacy/access rules where required.

---

## Production readiness by plane

Each modality plane **MUST** state maturity **separately** using [`MATURITY_TAXONOMY.md`](../technical/guides/MATURITY_TAXONOMY.md). A strong score on one plane does **not** imply readiness on another. Undifferentiated "modality is production-ready" claims are **invalid**.

## Modality Maturity Statement

- Plane A — Generative multimodal LLM:
  - Architecture maturity: A4
  - Implementation maturity: I3
  - Production readiness: P2
  - Evidence maturity: E3
- Plane B — Media/document ingest:
  - Architecture maturity: A4
  - Implementation maturity: I3
  - Production readiness: P2
  - Evidence maturity: E3
- Plane C — Dedicated inference / CV / ML:
  - Architecture maturity: A4
  - Implementation maturity: I2
  - Production readiness: P1
  - Evidence maturity: E2
- Notes:
  - Plane-specific maturity — not a single headline label for "modality."
  - Plane B ingest paths (Whisper, parsers, embeddings) are lab-stable; production limits vary by parser slug and tenant policy.
  - Plane C remote serving (`triton`, `huggingface_inference`) is wired; in-process heavy models require `ModalityExecutionProfile` and are not default production paths.
  - Update this block when any plane changes; cross-ref [`plan/MODALITY.md`](../maintainers/plans/MODALITY.md) for delivery evidence.

---
