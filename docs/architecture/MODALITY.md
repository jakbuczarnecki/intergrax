# Intergrax — Model & Modality Plane

**Status:** Canonical architecture document
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Implementation:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md) · [`plan/`](../plan/)

---


**Last updated:** 2026-06-02 · **Phase W-ML** (W-ML.0–W-ML.8 **Done**; harness backends: OpenCV contours, optional Ultralytics, ElevenLabs TTS when keyed; lab `ModalityProfile` wiring **Done**)

Catalog and harness rules for **vision**, **audio/speech**, **classical ML**, and **Hugging Face** usage — aligned with Integration → Tool → Skill → Agent (§5.3, §7.1.9).

**Canon:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.9 · §5.3  
**Target model:** [IDEAL_HARNESS_AI_ARCHITECTURE.md](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5.1, §7.1  
**Implementation tracker:** [intergrax_runtime_architecture.md](plan/MODALITY.md) **Phase W-ML**  
**Related:** [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md) (generative multimodal) · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) (catalog slugs) · RAG §7.1.2 (ingest/embeddings)

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

## Integration categories (planned extensions)

New categories require §5.2.4 approval. Planned slugs are **documentation placeholders** until Phase W-ML registers them.

| Category | Contract (planned) | Example slugs (planned) |
|----------|-------------------|-------------------------|
| **speech_provider** | TTS/STT SaaS API | `elevenlabs`, `azure_speech`, `deepgram`, `openai_tts` |
| **vision_serving** | Remote CV server gRPC/REST | `triton`, `torchserve`, `roboflow` |
| **ml_inference_host** | Managed model endpoint | `huggingface_inference`, `sagemaker`, `azure_ml`, `vertex_prediction` |

**Existing:** `document_parser` (ingest), `rerank_provider`, observability (`wandb`, `arize`, `phoenix`) for **eval**, not training.

---

## Tool surface

Atomic tools (LLM-selectable, MCP-exportable):

| tool_id | Plane | Status |
|---------|-------|--------|
| `vision.detect` | C | **Done** |
| `vision.segment` | C | **Done** |
| `vision.ocr_regions` | C | **Done** |
| `speech.synthesize` | C + speech_provider | **Done** — `IntegrationSpeechAdapter` bridges catalog `speech_provider` slugs (`elevenlabs`, `deepgram`) into Tier-0 speech tools via `wire_integration_tool_context()` |
| `speech.transcribe` | B or speech_provider | **Done** (stub / provider) |
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

## Declarative profiles (mirrors `LLMProfile`)

| Profile | Module | Factory |
|---------|--------|---------|
| **Vision** | `intergrax.model_inference.registry.VisionProfile` | `create_adapter()` → `VisionInferenceAdapter`; `build_registry()` → `ModelInferenceRegistry` |
| **Speech** | `intergrax.speech_adapters.SpeechProfile` | `create_adapter()` → `SpeechAdapter` |
| **Execution** | `intergrax.model_inference.execution.ModalityExecutionProfile` | `build_modality_inference_executor()` → `ModalityInferenceExecutor` (`in_process`, `thread_pool`, `celery`) |

**Execution env (harness):**

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_MODALITY_EXECUTION` | `in_process` \| `thread_pool` \| `celery` |
| `INTERGRAX_MODALITY_EXECUTION_WORKERS` | Thread pool size (default `4`) |
| `INTERGRAX_MODALITY_CELERY_BROKER_URL` | Celery broker for distributed modality jobs |
| `INTERGRAX_MODALITY_CELERY_EAGER` | `true` runs Celery tasks in-process (tests) |

Example (application host)::

```python
from intergrax.model_inference.registry import VisionProfile, VisionProvider, vision_profile_from_env
from intergrax.speech_adapters import SpeechProfile, SpeechProvider, speech_profile_from_env

vision = VisionProfile(provider=VisionProvider.OPENCV)
registry = vision.build_registry()

speech = speech_profile_from_env()
adapter = speech.create_adapter()
```

Registries: `VisionAdapterRegistry`, `SpeechAdapterRegistry` (same pattern as `LLMAdapterRegistry`).

## Harness environment variables

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_VISION_PROVIDER` or legacy `INTERGRAX_VISION_ADAPTER` | `stub` \| `onnxruntime` (OpenCV, default) \| `yolo_ultralytics` |
| `INTERGRAX_VISION_ARTIFACT_ID` | Optional artifact override for `vision.detect` default |
| `INTERGRAX_SPEECH_PROVIDER` | `stub` \| `elevenlabs` (default `stub`, or `elevenlabs` when `ELEVENLABS_API_KEY` set) |
| `INTERGRAX_SPEECH_VOICE_ID` | Optional default TTS voice |
| `ELEVENLABS_API_KEY` | API key for `SpeechProfile(provider=elevenlabs)` |
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
| `speech.synthesize` / `speech.transcribe` tools + ElevenLabs/stub backend | **Done** (W-ML.2) |
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
