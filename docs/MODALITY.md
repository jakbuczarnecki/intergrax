# Intergrax — Model & Modality Plane

**Last updated:** 2026-06-02 · **Phase W-ML** (W-ML.0 docs **Done**; W-ML.1–W-ML.8 harness contracts **Done**; production provider backends **incremental**)

Catalog and harness rules for **vision**, **audio/speech**, **classical ML**, and **Hugging Face** usage — aligned with Integration → Tool → Skill → Agent (§5.3, §7.1.9).

**Canon:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.9 · §5.3  
**Target model:** [IDEAL_HARNESS_AI_ARCHITECTURE.md](IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5.1, §7.1  
**Implementation tracker:** [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase W-ML**  
**Related:** [LLM_ADAPTERS.md](LLM_ADAPTERS.md) (generative multimodal) · [INTEGRATIONS.md](INTEGRATIONS.md) (catalog slugs) · RAG §7.1.2 (ingest/embeddings)

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
- **Worker pool** — default for GPU-heavy YOLO; queue via existing `message_bus` (Celery/Kafka) when Tier-3 enables it.
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

## Tool surface (planned)

Atomic tools (LLM-selectable, MCP-exportable):

| tool_id (planned) | Plane | Notes |
|-------------------|-------|-------|
| `vision.detect` | C | Bounding boxes / classes; policy on confidence threshold |
| `vision.segment` | C | Masks / polygons |
| `vision.ocr_regions` | C | Layout OCR when LLM vision insufficient |
| `speech.synthesize` | C + speech_provider | TTS output → `object_storage` URI |
| `speech.transcribe` | B or speech_provider | Prefer `whisper` parser for ingest |
| `ml.predict` | C | Tabular / vector in → structured out |
| `ml.explain` | C | Optional SHAP-like; high risk tier |

Skills MAY bundle these `tool_ids` (e.g. `harness.vision_qa`) — skills are not new inference engines.

---

## Agent assembly — ModalityProfile (planned)

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
| CV / ML inference (planned) | `modality_metrics` on `TASK_COMPLETED`: `inference_latency_ms`, `model_slug`, `device`, `batch_size` |
| Budgets | Extend V-COST: `inference_ms`, `media_bytes`, `tts_characters` |

---

## Explicit non-goals (Harness scope boundary)

- Online training / AutoML inside Nexus
- Feature store as platform product
- Replacing MLOps teams’ experiment tracking (use `wandb` integration for **eval linkage** only)
- CV models as `ToolContract` blobs without schema (no “mega tools”)
- Importing `torch` / `ultralytics` in Tier-2 `agents/`

---

## Implementation status (summary)

| Item | Status |
|------|--------|
| Architecture & catalog (this doc + canon §7.1.9) | **Done** (2026-06-02) |
| Whisper / yt_dlp / image ingest | **Done** (beta) |
| HF embeddings / optional SPLADE | **Done** |
| Multimodal LLM contract + attachment wire-up | **Planned** (W-ML.1) |
| `speech_provider` + tools | **Planned** (W-ML.2) |
| `model_inference` + YOLO/ONNX registry | **Planned** (W-ML.3) |
| HF Inference / Triton integration slugs | **Planned** (W-ML.4) |

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
[ ] 9. USAGE.md under provider folder; update INTEGRATIONS.md or this file
```

Agents: declare `tool_ids` / `ModalityProfile` — never import vendor SDKs.
