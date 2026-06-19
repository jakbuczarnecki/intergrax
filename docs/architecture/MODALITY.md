# Modality

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/MODALITY.md`](../plan/MODALITY.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 29  
**Audit instruction:** [`audit/MODALITY.md`](../audit/MODALITY.md)  
**Last updated:** 2026-06-19 — MOD-SPEECH-ARCH (speech slug identity; [ADR-MOD-001](../adr/entries/2026-06-19/ADR-MOD-001.md))

---

## Why this document exists

Harness AI at scale needs more than text LLMs: images, audio, dedicated CV detectors (YOLO, SAM, OCR pipelines), embeddings, rerankers, and batch classifiers. Intergrax already implements **parts** of this (Whisper ingest, HF embeddings, image smart loaders) without a single architectural name.

This file is the **modality index**: which plane owns what, how to extend it, and what agents may call.

---

## Three modality planes (canonical)

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

## Plane A — Generative multimodal LLM

- **Module:** `intergrax/llm_adapters/` only.
- **Capabilities (target contract):** `supports_vision`, `supports_audio_input`, `supports_audio_output` on `LLMAdapter`.
- **Messages:** `intergrax/llm/messages.py` — `AttachmentRef` (`image`, `audio`, `video`, …); adapters MUST map attachments to vendor content parts when capability flags are true.
- **When to use:** interactive reasoning, captioning in chat, tool planning with visual context.

**Do not** register OpenAI/Gemini/Claude as `integration` slugs.

---

## Plane B — Media ingest (RAG path)

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

## Plane C — Dedicated vision & ML inference

For **production CV** and **classical ML** where a multimodal LLM is the wrong tool (latency, cost, determinism, regulated bounding boxes).

### C.1 Vision inference engine (extensible)

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

### C.2 Classical ML (non-CV)

**Contract:** `ModelInferenceAdapter` — sklearn, XGBoost, ONNX classifiers, small torch models.

| Concern | Harness approach |
|---------|------------------|
| Artifact | `ModelArtifact` record: id, version, schema, owner, risk_tier, license |
| Invocation | Tool `ml.predict` / `ml.batch_predict` |
| Versioning | SemVer + immutable artifact URI (object storage) |
| Eval | Reuse V-EVAL + braintrust/phoenix/wandb observability slugs |

### C.3 Hugging Face — four roles (do not conflate)

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
| **speech_provider** | `SpeechProviderBackend` (TTS/STT SaaS) | `elevenlabs`, `deepgram` | Manifest + factory or `IntegrationPlugin` — **slug identity only** ([ADR-MOD-001](../adr/entries/2026-06-19/ADR-MOD-001.md)) |
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

## Tool surface

Atomic tools (LLM-selectable, MCP-exportable):

| tool_id | Plane | Status |
|---------|-------|--------|
| `vision.detect` | C | **Done** |
| `vision.segment` | C | **Done** |
| `vision.ocr_regions` | C | **Done** |
| `speech.synthesize` | C + `speech_provider` | **Done** — `SpeechProviderBackend` from Integration Library via `wire_integration_tool_context()` |
| `speech.transcribe` | C + `speech_provider` (or Plane B ingest) | **Done** — catalog slug (e.g. `deepgram`) or ingest parsers for files |
| `ml.predict` | C | **Done** |
| `ml.explain` | C | **Done** (feature importance stub) |
| `ml.batch_predict` | C | **Done** |

Skills MAY bundle these `tool_ids` (e.g. `harness.vision_qa`) — skills are not new inference engines.

---

## Agent assembly — ModalityProfile

Extend harness composition (canon §7.1.9, ideal §17):

```text
Agent = LLMProfile + ModalityProfile + Skill Set + Policy Bundle + Context Profile + Memory Profile + Tool Permissions
```

| ModalityProfile field | Purpose |
|-----------------------|---------|
| `allowed_planes` | `generative`, `ingest`, `vision_inference`, `classical_ml`, `speech` |
| `vision_model_ids` | Allowlist of registered CV models |
| `max_media_bytes` | Upload / attachment cap |
| `tts_voice_id` | Default voice for `speech.synthesize` |
| `require_deterministic_cv` | Force Plane C over Plane A for regulated domains |

### Three-plane ops runbook (MOD-MAINT-03)

| Plane | Operator action | When to use | Escalation |
|-------|-----------------|-------------|------------|
| **A — Generative** | Route via `ModalityProfile.allowed_planes` includes `generative`; monitor `llm_metrics` token/cost | Multimodal Q&A, captioning, unstructured media understanding | LLM adapter failover — [`LLM_ADAPTERS.md`](../plan/LLM_ADAPTERS.md) |
| **B — Ingest** | Use RAG/parser pipeline; never bypass `ParserPipeline` for prod ingest | Document/audio ingest to retrieval index | RAG ops — [`RAG.md`](../plan/RAG.md) §6.1av |
| **C — Deterministic CV/ML** | Set `require_deterministic_cv=true`; verify `opencv_runtime_available()` in runner; use harness registry artifacts | Regulated vision, golden-test CV, Celery modality jobs | MOD-MAINT OpenCV probe + `tests/unit/model_inference/` |

**Boundary rule:** Plane C outputs are tool-attributed (`modality_metrics`); Plane A outputs are LLM-attributed — do not mix cost attribution on a single step without explicit `ModalityProfile` plane selection.

**MOD-MAINT-04 backlog:** Triton/HF remote serving depth remains incremental post W-ML — register only; no online training scope.

---

## Observability & cost

| Signal | Mechanism |
|--------|-----------|
| LLM multimodal tokens | Existing `llm_metrics` |
| RAG ingest | `rag_metrics`, parser trace |
| CV / ML inference | Per-tool `modality_metrics` on `tool_invocation_end` from typed `ModalityInvocationCounters`; aggregated on `TASK_COMPLETED` when a `RunTraceReader` is wired; `export_run_metrics` uses the same aggregation |
| Speech | `tts_characters` and output `audio_uri` byte size recorded on `speech.synthesize` / `speech.transcribe` |
| Budgets | V-COST fields: `inference_ms`, `media_bytes`, `tts_characters`, `vision_detections`, `ml_predictions` |

---

## Explicit non-goals (Harness scope boundary)

- Online training / AutoML inside Nexus
- Feature store as platform product
- Replacing MLOps teams’ experiment tracking (use `wandb` integration for **eval linkage** only)
- CV models as `ToolContract` blobs without schema (no “mega tools”)
- Importing `torch` / `ultralytics` in Tier-2 `agents/`

---

## Declarative profiles (mirrors `LLMProfile` where applicable)

| Profile | Module | Factory / resolution |
|---------|--------|----------------------|
| **Vision** | `intergrax.model_inference.registry.VisionProfile` | `create_adapter()` → `VisionInferenceAdapter`; `build_registry()` → `ModelInferenceRegistry` |
| **Speech** | `IntegrationProfile.speech_provider` slot **or** `intergrax.speech_adapters.SpeechProfile` | `profile.resolve(SPEECH_PROVIDER)` → `SpeechProviderBackend`; env slug resolves against integration catalog ([ADR-MOD-001](../adr/entries/2026-06-19/ADR-MOD-001.md)) |
| **Execution** | `intergrax.model_inference.execution.ModalityExecutionProfile` | `build_modality_inference_executor()` → `ModalityInferenceExecutor` (`in_process`, `thread_pool`, `celery`) |

**Speech rule:** prefer `IntegrationProfile` for Tier-3 hosts. `SpeechProfile` is a thin env/lab helper over catalog slug resolution — not a parallel vendor registry.

**Execution env (harness):**

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_MODALITY_EXECUTION` | `in_process` \| `thread_pool` \| `celery` |
| `INTERGRAX_MODALITY_EXECUTION_WORKERS` | Thread pool size (default `4`) |
| `INTERGRAX_MODALITY_CELERY_BROKER_URL` | Celery broker for distributed modality jobs |
| `INTERGRAX_MODALITY_CELERY_EAGER` | `true` runs Celery tasks in-process (tests) |

Example (application host):

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.providers.speech_provider.deepgram.manifest import MANIFEST as DEEPGRAM
from intergrax.model_inference.registry import VisionProfile, VisionProvider

vision = VisionProfile(provider=VisionProvider.OPENCV)
registry = vision.build_registry()

integration = IntegrationProfile(speech_provider=DEEPGRAM)
speech_backend = integration.resolve(IntegrationCategory.SPEECH_PROVIDER)
```

Registries: `VisionAdapterRegistry` (vision slugs). Speech SaaS vendors use **Integration Library** catalog registration — not a closed enum.

## Harness environment variables

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_VISION_PROVIDER` or legacy `INTERGRAX_VISION_ADAPTER` | `stub` \| `onnxruntime` (OpenCV, default) \| `yolo_ultralytics` |
| `INTERGRAX_VISION_ARTIFACT_ID` | Optional artifact override for `vision.detect` default |
| `INTERGRAX_SPEECH_PROVIDER` | Integration catalog slug (e.g. `elevenlabs`, `deepgram`, `stub`) — resolved via `IntegrationProfile`, not enum |
| `INTERGRAX_SPEECH_VOICE_ID` | Optional default TTS voice for `speech.synthesize` |
| `INTERGRAX_ELEVENLABS_*` / `INTERGRAX_DEEPGRAM_*` | Per-slug integration env prefixes (see provider `USAGE.md`) |
| `INTERGRAX_TRITON_URL` | Triton/KServe base URL for `VisionProvider.TRITON` |
| `INTERGRAX_TRITON_MODEL` | Triton model name (default `yolo`) |
| `HUGGINGFACE_API_KEY` | HF Inference API for `VisionProvider.HUGGINGFACE_INFERENCE` |
| `INTERGRAX_HF_VISION_MODEL` | HF model id (default `facebook/detr-resnet-50`) |
| `LEGAL_ENABLE_MODALITY_TOOLS` | Enable Plane C tools on legal host with profile extras |
| `INTERGRAX_MODALITY_EXECUTION` | `in_process` (default) or `thread_pool` for heavy vision adapters |
| `INTERGRAX_MODALITY_EXECUTION_WORKERS` | Thread pool size (default `4`) |

## Implementation status (summary)

| Item | Status |
|------|--------|
| Architecture & catalog (this doc + canon §7.1.9) | **Done** (2026-06-02) |
| Whisper / yt_dlp / image ingest | **Done** (beta) |
| HF embeddings / optional SPLADE | **Done** |
| Multimodal LLM contract + attachment wire-up | **Done** (W-ML.1) |
| `speech.synthesize` / `speech.transcribe` tools + integration catalog slugs | **Done** (W-ML.2) · slug alignment **Done** (MOD-SPEECH-ARCH) |
| `model_inference` registry + OpenCV / stub / optional Ultralytics | **Done** (W-ML.3) |
| Lab `ModalityProfile` + `ToolAccessPolicy.apply_modality_profile` | **Done** (W-ML.6) |
| Golden fixture `tests/fixtures/vision_golden/sample_target.png` | **Done** |
| HF Inference / Triton live endpoints | **Placeholder** (W-ML.4) |

---

## Extension checklist (new vision backend)

```text
[ ] 1. ADR: plane C vs A vs B for the use case
[ ] 2. VisionInferenceAdapter implementation under model_inference/providers/<slug>/
[ ] 3. Register in VisionInferenceRegistry + VisionModelProfile defaults
[ ] 4. Optional: integration slug if remote-only (vision_serving / ml_inference_host)
[ ] 5. ToolContract(s) with JSON schema I/O — one atomic operation per tool_id
[ ] 6. Policy: risk_tier, max_batch, allowed MIME types, tenant allowlist
[ ] 7. Metrics + trace fields on ToolInvocation
[ ] 8. Unit tests (golden tensors or fixture images) + gate subset
[ ] 9. USAGE.md under provider folder; update architecture/INTEGRATIONS.md or this file
```

Agents: declare `tool_ids` / `ModalityProfile` — never import vendor SDKs.
