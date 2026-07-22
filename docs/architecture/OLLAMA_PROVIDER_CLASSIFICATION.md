# Ollama provider classification

**Status:** frozen (INTERACTIONS-TAXONOMY-1)  
**Date:** 2026-07-22

## Current behavior (pre-migration evidence)

The previous `interaction_surface/ollama` package wrapped an inbound
`InteractionAdapter` around a thin HTTP client to the native Ollama API
(`GET /api/tags`). Public operations were:

| Operation | Present | Notes |
|-----------|---------|-------|
| `health` | yes | `/api/tags` status |
| `list_models` | yes | parse `/api/tags` |
| `can_handle` / `to_inbound` | yes | misclassified intake bridge |
| model generation / chat | no | owned by `llm_adapters` |
| embeddings | no | owned by `rag.embedding` |
| model pull / lifecycle / process spawn | no | infra/compose only |

Docstring on the adapter: “Health/model-list surface for local Ollama host
(modality bridge, not LLM catalog).”

## Actual consumers

- Catalog registration / P5 M.6 P4 shell probes
- Compose / LKW stacks use Ollama as the local LLM host via `INTERGRAX_LLM_PROVIDER=ollama` and `llm_adapters` — **not** via this integration’s InteractionAdapter
- No application product path depended on Ollama as an intake surface

## Candidate categories evaluated

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| `interaction_surface` | rejected / removed | Not a vendor-substitutable intake category; Ollama is not an interaction protocol |
| `ml_inference_host` | rejected | Contract is managed remote `predict(model_ref, inputs)` (Replicate, planned HF Inference). Architecture text: “Managed model endpoint”. Ollama does not implement `predict` in this package |
| `speech_provider` | rejected | TTS/STT SaaS only |
| `vision_serving` | rejected | Remote CV servers (Triton); different modality |
| `workflow_orchestrator` / `sandbox_host` / `cloud_platform` | rejected | Unrelated operational roles |
| New `model_serving_runtime` | **selected** | Matches self-hosted model serving hosts |

## Selected category

```text
model_serving_runtime
→ ModelServingRuntimeIntegrationContract
→ ModelServingRuntimeBackend (list_models, health)
```

### Shared semantics

Application-facing operations for a self-hosted model serving host:

- probe host health
- list available model identifiers

Does **not** own LLM chat/completion or embeddings (those remain
`llm_adapters` / RAG providers).

### Natural substitute providers

Systems that would implement the same contract:

- vLLM
- llama.cpp server
- TGI (text-generation-inference)
- LocalAI
- SGLang

None of these packages exist in the repository today; they are documented as
semantic peers only.

## Contract sufficiency

Existing `MlInferenceHostIntegrationContract` was insufficient without forcing
a false `predict` surface. A new category was required to avoid overlapping
managed-endpoint semantics.

No change to `ml_inference_host` was required.

## Migration impact

- `ollama` → `intergrax/integrations/providers/model_serving_runtime/ollama/`
- Public class: `OllamaModelServingRuntimeIntegration`
- `SLUG_CATEGORY["ollama"] = "model_serving_runtime"`
- Inbound `OllamaInteractionAdapter` remains a private p5 helper for tests only;
  it is **not** a provider identity

## Explicitly deferred

- Generation / chat / embeddings through this integration
- Model pull / download / process lifecycle APIs
- OpenAI-compatible proxy surface as a first-class contract method
- Shipping vLLM / llama.cpp / TGI / LocalAI / SGLang provider packages
