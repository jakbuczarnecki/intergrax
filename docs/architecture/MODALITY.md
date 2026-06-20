# Modality

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/MODALITY.md`](../plan/MODALITY.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 29  
**Audit instruction:** [`audit/MODALITY.md`](../audit/MODALITY.md)  
**Last updated:** 2026-06-19 — MOD-SPEECH-ARCH (speech slug identity; [ADR-MOD-001](../adr/entries/2026-06-19/ADR-MOD-001.md))

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (MODALITY canon).

- **Implement / audit default:** vision/audio modality adapters. Skip modality inventory unless MOD task.
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/MODALITY.md`](../plan/MODALITY.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/MODALITY.md`](../guides/audit_slices/MODALITY.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`arch/MODALITY_tool_surface_detail.md`](arch/MODALITY_tool_surface_detail.md) | tool surface detail |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


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
