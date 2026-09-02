# MODALITY - tool surface detail

**Parent hub:** [`MODALITY.md`](../MODALITY.md)

## Tool surface

Atomic tools (LLM-selectable, MCP-exportable):

| tool_id | Plane | Status |
|---------|-------|--------|
| `vision.detect` | C | **Done** |
| `vision.segment` | C | **Done** |
| `vision.ocr_regions` | C | **Done** |
| `speech.synthesize` | C + `speech_provider` | **Done** - `SpeechProviderBackend` from Integration Library via `wire_integration_tool_context()` |
| `speech.transcribe` | C + `speech_provider` (or Plane B ingest) | **Done** - catalog slug (e.g. `deepgram`) or ingest parsers for files |
| `ml.predict` | C | **Done** |
| `ml.explain` | C | **Done** (feature importance stub) |
| `ml.batch_predict` | C | **Done** |

Skills MAY bundle these `tool_ids` (e.g. `harness.vision_qa`) - skills are not new inference engines.

---

## Agent assembly - ModalityProfile

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
| **A - Generative** | Route via `ModalityProfile.allowed_planes` includes `generative`; monitor `llm_metrics` token/cost | Multimodal Q&A, captioning, unstructured media understanding | LLM adapter failover - [`LLM_ADAPTERS.md`](../plan/LLM_ADAPTERS.md) |
| **B - Ingest** | Use RAG/parser pipeline; never bypass `ParserPipeline` for prod ingest | Document/audio ingest to retrieval index | RAG ops - [`RAG.md`](../plan/RAG.md) §6.1av |
| **C - Deterministic CV/ML** | Set `require_deterministic_cv=true`; verify `opencv_runtime_available()` in runner; use harness registry artifacts | Regulated vision, golden-test CV, Celery modality jobs | MOD-MAINT OpenCV probe + `tests/unit/model_inference` |

**Boundary rule:** Plane C outputs are tool-attributed (`modality_metrics`); Plane A outputs are LLM-attributed - do not mix cost attribution on a single step without explicit `ModalityProfile` plane selection.

**MOD-MAINT-04 backlog:** Triton/HF remote serving depth remains incremental post W-ML - register only; no online training scope.

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
- Importing `torch` / `ultralytics` in Tier-2 `agents`

---

## Declarative profiles (mirrors `LLMProfile` where applicable)

| Profile | Module | Factory / resolution |
|---------|--------|----------------------|
| **Vision** | `intergrax.model_inference.registry.VisionProfile` | `create_adapter()` → `VisionInferenceAdapter`; `build_registry()` → `ModelInferenceRegistry` |
| **Speech** | `IntegrationProfile.speech_provider` slot **or** `intergrax.speech_adapters.SpeechProfile` | `profile.resolve(SPEECH_PROVIDER)` → `SpeechProviderBackend`; env slug resolves against integration catalog ([ADR-MOD-001](../adr/entries/2026-06-19/ADR-MOD-001.md)) |
| **Execution** | `intergrax.model_inference.execution.ModalityExecutionProfile` | `build_modality_inference_executor()` → `ModalityInferenceExecutor` (`in_process`, `thread_pool`, `celery`) |

**Speech rule:** prefer `IntegrationProfile` for Tier-3 hosts. `SpeechProfile` is a thin env/lab helper over catalog slug resolution - not a parallel vendor registry.

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

Registries: `VisionAdapterRegistry` (vision slugs). Speech SaaS vendors use **Integration Library** catalog registration - not a closed enum.

## Harness environment variables

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_VISION_PROVIDER` or legacy `INTERGRAX_VISION_ADAPTER` | `stub` \| `onnxruntime` (OpenCV, default) \| `yolo_ultralytics` |
| `INTERGRAX_VISION_ARTIFACT_ID` | Optional artifact override for `vision.detect` default |
| `INTERGRAX_SPEECH_PROVIDER` | Integration catalog slug (e.g. `elevenlabs`, `deepgram`, `stub`) - resolved via `IntegrationProfile`, not enum |
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
[ ] 5. ToolContract(s) with JSON schema I/O - one atomic operation per tool_id
[ ] 6. Policy: risk_tier, max_batch, allowed MIME types, tenant allowlist
[ ] 7. Metrics + trace fields on ToolInvocation
[ ] 8. Unit tests (golden tensors or fixture images) + gate subset
[ ] 9. USAGE.md under provider folder; update architecture/INTEGRATIONS.md or this file
```

Agents: declare `tool_ids` / `ModalityProfile` - never import vendor SDKs.
