# Implementation Phases — Llm And Modality

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Phase W-ML — Model & Modality Plane (Vision, Audio, Classical ML)

**Status:** **Done** (2026-06-02) — docs + implementation waves W-ML.0–W-ML.8.  
**Canon:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.9, §53.13 · **Catalog:** [`architecture/MODALITY.md`](architecture/MODALITY.md) · **Ideal:** [IDEAL_HARNESS_AI_ARCHITECTURE.md](IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5.1, §7.1, §17.

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

