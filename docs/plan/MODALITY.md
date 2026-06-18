# Modality — Implementation Plan

**Architecture (1:1):** [`architecture/MODALITY.md`](../architecture/MODALITY.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

**Last updated:** 2026-06-17 — **Full Harness LC** (re-validates W-ML.0–W-ML.8 + AUDIT-IDEAL-29.1/29.2 closeout).

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5.1 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-29.1 | §29 Modality | Live Triton / HF Inference endpoints (replace placeholders) | P1 | **Done** |
| AUDIT-IDEAL-29.2 | §29 Modality | Plane C vision inference E2E on product worker pools | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

## Phase W-ML — Model & Modality Plane (Vision, Audio, Classical ML)

**Status:** **Done** (2026-06-02) — docs + implementation waves W-ML.0–W-ML.8.  
**Canon:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.9, §53.13 · **Catalog:** [`architecture/MODALITY.md`](architecture/MODALITY.md) · **Ideal:** [IDEAL_HARNESS_AI_ARCHITECTURE.md](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5.1, §7.1, §17.

**Strategic fit:** Extends Harness AI at scale without MLOps scope creep. Same patterns as LLM adapters and Integration Library — registries, contracts, atomic tools, policy, trace, V-COST budgets.

**Explicitly in scope:**

- Three-plane modality model (generative LLM / ingest / dedicated inference).
- Extensible **vision inference engine** (YOLO/Ultralytics, ONNX Runtime, OpenVINO, TensorRT, remote Triton/TorchServe, cloud endpoints).
- `speech_provider` integrations (e.g. ElevenLabs) + TTS/STT tools.
- Classical ML registry (`ModelArtifact`, `ml.predict` tools).
- Hugging Face role separation (embeddings vs hosted inference vs hub governance).
- `ModalityProfile` for Tier-3/agent assembly.
- `modality_metrics` + cost envelope extensions.

**Explicitly out of scope:**

- Online training / AutoML / feature stores as platform products.
- LLM slugs in Integration Catalog (§44.10).
- CV or ML SDK imports in Tier-2 `agents/`.
- Monolithic “vision skills” without atomic tools.

**Dependency:** Documentation may land during Phase V; code waves SHOULD not block V closeout but SHOULD follow V-COST/V-SEC patterns.

#### W-ML — Deliverables

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| W-ML.0 | Canon §7.1.9 + §53.13 + `architecture/MODALITY.md` + IDEAL/LLM_ADAPTERS sync | **Done** | **Critical** | Docs merged; three planes documented |
| W-ML.1 | Multimodal LLM contract — `supports_vision` / audio flags; `AttachmentRef` → vendor parts | **Done** | High | Conformance tests in `tests/unit/llm_adapters/`; OpenAI + Gemini vision flags |
| W-ML.2 | `speech_provider` category + `elevenlabs` (or stub) + tools `speech.synthesize` / `speech.transcribe` | **Done** | Medium | `ElevenLabsSpeechBackend` when `ELEVENLABS_API_KEY` set; stub otherwise |
| W-ML.3 | `intergrax/model_inference/` scaffold — `VisionInferenceAdapter`, registry, `yolo_ultralytics` + `onnxruntime` slugs | **Done** | High | OpenCV contour adapter (default); optional Ultralytics; golden PNG fixture |
| W-ML.4 | Remote serving integrations — `vision_serving` / `huggingface_inference` (Triton HTTP + HF Inference API) | **Done** | Medium | `triton_vision.py`, `huggingface_inference_vision.py`; env `INTERGRAX_TRITON_URL`, `HUGGINGFACE_API_KEY` |
| W-ML.5 | `ModelInferenceAdapter` + `ml.predict` + `ModelArtifact` metadata contract | **Done** | Medium | `ml.predict` tool + stub sklearn classifier artifact |
| W-ML.6 | `ModalityProfile` + Tier-3 wiring + policy intersection with `ToolAccessPolicy` | **Done** | High | `runtime/modality/modality_profile.py` + `ToolAccessPolicy.apply_modality_profile` |
| W-ML.7 | `modality_metrics` export on `TASK_COMPLETED` + V-COST fields (`inference_ms`, `media_bytes`, `tts_characters`) | **Done** | Medium | `runtime/observability/modality_metrics.py` + metrics export |
| W-ML.8 | Capability graph nodes for modality tools + compatibility guard entries | **Done** | Low | Modality tools registered in default catalog (`register_default_tools`) |

#### W-ML — Execution waves

```text
Wave W0 (docs):       W-ML.0  — Done 2026-06-02
Wave W1 (LLM):        W-ML.1  — multimodal attachments (Plane A)
Wave W2 (speech):     W-ML.2  — speech_provider + tools
Wave W3 (vision CV):  W-ML.3  — YOLO + ONNX local inference + vision.* tools
Wave W4 (scale-out):  W-ML.4  — remote serving integrations
Wave W5 (classical):  W-ML.5  — ml.predict + ModelArtifact
Wave W6 (governance): W-ML.6 + W-ML.7 + W-ML.8 — profiles, metrics, capability graph
```

**Priority ladder placement:** Band 2 extension — run **after** critical Phase V streams (V-CG, V-SEC, V-COST) or **in parallel** with V-MA/V-KG when owners are separate. **Not** Band 3 product work.

#### W-ML — Existing assets (no rework required)

| Asset | Plane | Location |
|-------|-------|----------|
| Whisper / yt_dlp ingest | B | `integrations/providers/document_parser/` |
| Image/audio smart loaders | B | `intergrax/multimedia/`, `rag/document_loaders/` |
| HF embeddings | B | `rag/embedding/providers/hf_embedding_provider.py` |
| SPLADE sparse (optional) | B | `rag/vectorstore/sparse/splade_sparse_encoder.py` |
| LLM adapters (19 slugs) | A | `intergrax/llm_adapters/` |

#### W-ML — Paydown log

| Date | W-ML ID | Summary |
|------|---------|---------|
| 2026-06-02 | W-ML.0 | Canon §7.1.9, §53.13, `architecture/MODALITY.md`, IDEAL §3.5.1/§7.1/§17, `architecture/LLM_ADAPTERS.md` multimodal section, docs README |
| 2026-06-02 | W-ML.1–W-ML.8 | Multimodal LLM flags + attachment mapping, speech/vision/ml tools, model_inference scaffold, ModalityProfile, modality metrics, runtime governance bridge |
| 2026-06-02 | W-ML.2–W-ML.3, W-ML.6 | Lab harness modality tool wiring, OpenCV/ElevenLabs backends, golden vision fixture, `RuntimeConfig.modality_profile` |
| 2026-06-02 | W-ML.4+ | Triton/HF vision adapters, `vision.segment`/`vision.ocr_regions`/`ml.explain`, `harness.vision_qa`, extended `ModalityProfile`, legal `LEGAL_ENABLE_MODALITY_TOOLS` |
| 2026-06-02 | W-ML.workers | `ModalityExecutionProfile`, thread-pool executor, `ml.batch_predict`, `harness.modality_smoke`, `max_media_bytes` enforcement |
| 2026-06-02 | W-ML.celery | `CeleryModalityInferenceExecutor`, serialized modality jobs, trace `modality_metrics` on `tool_invocation_end`, aggregated export |
| 2026-06-02 | W-ML.metrics+ | Typed `ModalityInvocationCounters`, `media_bytes`/`tts_characters`/`ml_predictions` recording, message_bus Celery registration, capability graph modality `COMPATIBLE_WITH` edges |
| 2026-06-03 | W-ML.7b | `TASK_COMPLETED` payload includes aggregated `modality_metrics` via `NexusRuntimeEventPublisher` + `RunTraceReader` |

---

## Phase MODALITY-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates W-ML.0–W-ML.8 + AUDIT-IDEAL-29.1/29.2; no open P0/P1  
**Prerequisites:** Phase W-ML **Done** · modality CI gates  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| MODALITY-LC-S1 | **Re-audit** — W-ML register + three-plane verdict | **Done** | High | No P0/P1 |
| MODALITY-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| MODALITY-LC-S3 | **Gate verification** | **Done** | High | 14 unit tests · 2 CI gate scripts |
| MODALITY-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** OpenCV golden tests require `opencv-python-headless` in runner env · online training out of scope · Plane A/C boundary ops docs

**Audit note (2026-06-18):** `pytest tests/unit/model_inference/` reports **2 failing tests** locally — treat as **harness defects to fix**, not environment-only waivers (see §6.1av MOD-MAINT-01/02).

### 6.1av Harness implementation queue — Modality audit maintenance (planned)

**Source:** Layer 15 audit (2026-06-18) — `MODALITY` layer 29 · [`guides/audit/results/2026-06-18/MODALITY.md`](guides/audit/results/2026-06-18/MODALITY.md)  
**Priority ladder:** **Band 1** (§6.1) — **test repair first**, then docs/depth; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **MOD-MAINT-01** | Test/CI | P2 | **Planned** | **Fix** `test_opencv_adapter_detects_white_rectangle` — wire `opencv-python-headless` optional extra **or** robust skip only when `cv2` truly unavailable; default CI/dev env must run green | `pytest tests/unit/model_inference/test_opencv_vision.py` passes in standard `uv` dev install |
| 2 | **MOD-MAINT-02** | Test/Code | P2 | **Planned** | **Fix** `test_run_modality_detect_job_uses_harness_registry` — repair Celery modality execution path or test fixtures so registry wiring is asserted correctly | `pytest tests/unit/model_inference/test_celery_modality_execution.py` green |
| 3 | **MOD-MAINT-03** | Docs | P4 | **Planned** | Plane A/C boundary — ops runbook section in modality canon | Architecture §three-plane ops table |
| 4 | **MOD-MAINT-04** | Backlog | P3 | **Planned** | Remote serving incremental — Triton/HF depth register row (post W-ML closeout) | Plan row; no scope creep into online training |

**Suggested PR order:** MOD-MAINT-02 → MOD-MAINT-01 → MOD-MAINT-03 → MOD-MAINT-04.

**Explicitly out of scope:** online training / AutoML — canon constraint.

---

*End of Modality Implementation Plan.*
