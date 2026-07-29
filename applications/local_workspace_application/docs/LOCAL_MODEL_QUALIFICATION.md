<!--
GENERATED FILE.
Do not edit benchmark results manually.
Regenerate with:
uv run python applications/local_workspace_application/scripts/run-local-model-qualification.py
-->

# LKW Local Ollama Model Qualification

## 1. Scope and interpretation

This is an LKW-specific benchmark for conversational interaction planning. It is not a universal LLM ranking. Results apply to the exact model tags, digests, Ollama version, configuration and benchmark host shown below.

## 2. Executive summary

No tested model/protocol pair met the full LKW qualification threshold.

- Required models: 5
- Provisioned models: 5
- Expected model/protocol pairs: 10
- Attempted model/protocol pairs: 10
- Expected scored calls: 360
- Actual scored calls: 144

## 3. Recommended configuration

No tested model/protocol pair met the full LKW qualification threshold.

## 4. Benchmark methodology

- Corpus version: `lkw.local_model_qualification.corpus.v1`
- Production semantic prompt via `build_planning_messages()`
- Structured output remains the current production transport
- `single_plan_tool` is a benchmark candidate only
- Submission tool `submit_conversation_interaction_draft` does not execute operations
- Repair is disabled in benchmark scoring (`repair_attempts=0`)
- Tool protocol uses `tool_choice=auto` with deterministic exactly-one-call enforcement by the benchmark harness
- Tool protocol adds only this transport instruction after the production system message:
  Call submit_conversation_interaction_draft exactly once with the complete semantic draft as its arguments. This submission tool does not execute any operation. Do not answer in plain text and do not call any other tool.

## Docker Ollama provisioning

- Runtime: `docker`
- Compose file: `../../../infra/docker/ollama/docker-compose.yml`
- Compose service: `ollama`
- Container name: `intergrax-ollama`
- Persistent model volume: `intergrax-ollama-models`
- Required model count: 5
- Docker Ollama readiness: READY

| Model | Provisioning status | Digest | Artifact size |
| --- | --- | --- | --- |
| qwen2.5:14b | ALREADY_AVAILABLE | `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` | 8.4 GiB |
| qwen3:14b | PULLED | `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8` | 8.6 GiB |
| llama3.1:8b | PULLED | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` | 4.6 GiB |
| gpt-oss:20b | PULLED | `17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7` | 12.8 GiB |
| mistral-small3.2:24b | PULLED | `5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b` | 14.1 GiB |

## 5. Benchmark host

- OS: Windows 11
- Architecture: AMD64
- Python: 3.12.4
- CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
- RAM: 63.7 GiB
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU
- GPU VRAM: 12.0 GiB
- NVIDIA driver: 581.32

Hardware figures are observed on this benchmark host, not universal minimum requirements.

## 6. Ollama environment

- Host: `http://localhost:11434`
- Version: 0.30.10

## 7. Tested model inventory

| Model | Role | Installed | Digest | Artifact size |
| --- | --- | --- | --- | --- |
| qwen2.5:14b | baseline | True | `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` | 8.4 GiB |
| qwen3:14b | same-size-generation-candidate | True | `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8` | 8.6 GiB |
| llama3.1:8b | lightweight-control | True | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` | 4.6 GiB |
| gpt-oss:20b | agentic-candidate | True | `17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7` | 12.8 GiB |
| mistral-small3.2:24b | instruction-following-candidate | True | `5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b` | 14.1 GiB |

## 8. Model × protocol comparison

| Model | Role | Protocol | Capabilities | Schema probe | Probe failure category | Probe phase | Safe error code | Samples | Semantic success | Invalid drafts | Provider failures | Unsafe state changes | Median latency | p95 latency | Execution mode | Qualification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:14b | baseline | structured_output | completion, tools | PASS | n/a | n/a | n/a | 36 | 13.9% | 0 | 31 | 0 | 21 | 40905 | FULL_GPU (measured) | NOT_QUALIFIED |
| qwen2.5:14b | baseline | single_plan_tool | completion, tools | SCHEMA_INCOMPATIBLE | MISSING_PLAN_TOOL_CALL | TOOL_CALL_VALIDATION | n/a | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | FULL_GPU (measured) | SCHEMA_INCOMPATIBLE |
| qwen3:14b | same-size-generation-candidate | structured_output | completion, thinking, tools | PASS | n/a | n/a | n/a | 36 | 50.0% | 1 | 0 | 1 | 17567 | 107538 | FULL_GPU (measured) | NOT_QUALIFIED |
| qwen3:14b | same-size-generation-candidate | single_plan_tool | completion, thinking, tools | SCHEMA_INCOMPATIBLE | MISSING_PLAN_TOOL_CALL | TOOL_CALL_VALIDATION | n/a | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | FULL_GPU (measured) | SCHEMA_INCOMPATIBLE |
| llama3.1:8b | lightweight-control | structured_output | n/a | PASS | n/a | n/a | n/a | 36 | 30.6% | 0 | 0 | 11 | 2650 | 6262 | FULL_GPU (measured) | NOT_QUALIFIED |
| llama3.1:8b | lightweight-control | single_plan_tool | n/a | SCHEMA_INCOMPATIBLE | MISSING_PLAN_TOOL_CALL | TOOL_CALL_VALIDATION | n/a | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | FULL_GPU (measured) | SCHEMA_INCOMPATIBLE |
| gpt-oss:20b | agentic-candidate | structured_output | completion, thinking, tools | PASS | n/a | n/a | n/a | 36 | 52.8% | 0 | 0 | 0 | 20066 | 40863 | UNKNOWN | NOT_QUALIFIED |
| gpt-oss:20b | agentic-candidate | single_plan_tool | completion, thinking, tools | SCHEMA_INCOMPATIBLE | DRAFT_VALIDATION_FAILED | DRAFT_VALIDATION | n/a | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | UNKNOWN | SCHEMA_INCOMPATIBLE |
| mistral-small3.2:24b | instruction-following-candidate | structured_output | n/a | PROVIDER_ERROR | PROVIDER_ERROR | PROVIDER_INVOKE | UNKNOWN_PROVIDER_FAILURE | 0 | 0.0% | 0 | 1 | 0 | 0 | 0 | UNKNOWN | PROVIDER_ERROR |
| mistral-small3.2:24b | instruction-following-candidate | single_plan_tool | n/a | PROTOCOL_UNSUPPORTED | PROTOCOL_UNSUPPORTED | CAPABILITY_CHECK | OLLAMA_MODEL_TOOLS_UNSUPPORTED | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | UNKNOWN | PROTOCOL_UNSUPPORTED |

## 9. Safety and state-change results

- qwen2.5:14b / structured_output: unsafe state changes = 0
- qwen2.5:14b / single_plan_tool: unsafe state changes = 0
- qwen3:14b / structured_output: unsafe state changes = 1
- qwen3:14b / single_plan_tool: unsafe state changes = 0
- llama3.1:8b / structured_output: unsafe state changes = 11
- llama3.1:8b / single_plan_tool: unsafe state changes = 0
- gpt-oss:20b / structured_output: unsafe state changes = 0
- gpt-oss:20b / single_plan_tool: unsafe state changes = 0
- mistral-small3.2:24b / structured_output: unsafe state changes = 0
- mistral-small3.2:24b / single_plan_tool: unsafe state changes = 0

## 10. Failure categories

### qwen2.5:14b / structured_output
- PROVIDER_ERROR: 31

### qwen2.5:14b / single_plan_tool
- MISSING_PLAN_TOOL_CALL: 1

### qwen3:14b / structured_output
- CANONICAL_VALIDATION_FAILED: 9
- DRAFT_COMPILATION_FAILED: 7
- DRAFT_VALIDATION_FAILED: 1
- UNNECESSARY_WORKSPACE_ACTIVATE: 1
- WRONG_ACTION_TYPE: 1

### qwen3:14b / single_plan_tool
- MISSING_PLAN_TOOL_CALL: 1

### llama3.1:8b / structured_output
- CANONICAL_VALIDATION_FAILED: 1
- DRAFT_COMPILATION_FAILED: 9
- MISSING_CLARIFICATION: 3
- UNEXPECTED_STATE_CHANGE: 4
- UNNECESSARY_CLARIFICATION: 1
- UNNECESSARY_WORKSPACE_ACTIVATE: 7
- WRONG_ACTION_COUNT: 5
- WRONG_ACTION_TYPE: 5
- WRONG_CANDIDATE_REFERENCE: 2
- WRONG_WORKSPACE_REFERENCE: 7

### llama3.1:8b / single_plan_tool
- MISSING_PLAN_TOOL_CALL: 1

### gpt-oss:20b / structured_output
- CANONICAL_VALIDATION_FAILED: 5
- DRAFT_COMPILATION_FAILED: 7
- WRONG_ACTION_TYPE: 5
- WRONG_WORKSPACE_REFERENCE: 3

### gpt-oss:20b / single_plan_tool
- DRAFT_VALIDATION_FAILED: 1

### mistral-small3.2:24b / structured_output
- PROVIDER_ERROR: 1

### mistral-small3.2:24b / single_plan_tool
- PROTOCOL_UNSUPPORTED: 1


## 11. Per-model details

### qwen2.5:14b

- Digest: `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`
- Artifact size: 8.4 GiB
- Parameter size: 14.8B
- Quantization: Q4_K_M
- Declared capabilities: completion, tools
- Observed loaded size: 8.8 GiB
- Observed VRAM allocation: 8.8 GiB
- Observed offload mode: FULL_GPU (measured from Client.ps())

#### Protocol: structured_output
- Qualification: NOT_QUALIFIED
- Top failure categories:
  - PROVIDER_ERROR: 31
- Failed case IDs: planner.active_workspace_source_add, planner.ambiguous_missing_workspace_target, planner.attachment_ingestion, planner.explicit_workspace_activation, planner.explicit_workspace_delete, planner.source_candidate_attach_ordinal, planner.source_candidate_list, planner.source_list_named_workspace, planner.target_workspace_without_activation, planner.url_question_not_ingestion, planner.workspace_list

#### Protocol: single_plan_tool
- Qualification: SCHEMA_INCOMPATIBLE
- Top failure categories:
  - MISSING_PLAN_TOOL_CALL: 1

### qwen3:14b

- Digest: `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`
- Artifact size: 8.6 GiB
- Parameter size: 14.8B
- Quantization: Q4_K_M
- Declared capabilities: completion, thinking, tools
- Observed loaded size: 9.0 GiB
- Observed VRAM allocation: 9.0 GiB
- Observed offload mode: FULL_GPU (measured from Client.ps())

#### Protocol: structured_output
- Qualification: NOT_QUALIFIED
- Top failure categories:
  - CANONICAL_VALIDATION_FAILED: 9
  - DRAFT_COMPILATION_FAILED: 7
  - DRAFT_VALIDATION_FAILED: 1
  - UNNECESSARY_WORKSPACE_ACTIVATE: 1
  - WRONG_ACTION_TYPE: 1
- Failed case IDs: planner.active_workspace_source_add, planner.ambiguous_missing_workspace_target, planner.attachment_ingestion, planner.mixed_source_ordinal_routing, planner.source_candidate_attach_ordinal, planner.source_candidate_list, planner.source_list_named_workspace

#### Protocol: single_plan_tool
- Qualification: SCHEMA_INCOMPATIBLE
- Top failure categories:
  - MISSING_PLAN_TOOL_CALL: 1

### llama3.1:8b

- Digest: `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`
- Artifact size: 4.6 GiB
- Parameter size: 8.0B
- Quantization: Q4_K_M
- Declared capabilities: n/a
- Observed loaded size: 4.9 GiB
- Observed VRAM allocation: 4.9 GiB
- Observed offload mode: FULL_GPU (measured from Client.ps())

#### Protocol: structured_output
- Qualification: NOT_QUALIFIED
- Top failure categories:
  - DRAFT_COMPILATION_FAILED: 9
  - UNNECESSARY_WORKSPACE_ACTIVATE: 7
  - WRONG_WORKSPACE_REFERENCE: 7
  - WRONG_ACTION_COUNT: 5
  - WRONG_ACTION_TYPE: 5
- Failed case IDs: planner.active_workspace_source_add, planner.ambiguous_missing_workspace_target, planner.attachment_ingestion, planner.mixed_source_ordinal_routing, planner.source_candidate_attach_ordinal, planner.source_candidate_list, planner.source_list_named_workspace, planner.target_workspace_without_activation, planner.url_question_not_ingestion

#### Protocol: single_plan_tool
- Qualification: SCHEMA_INCOMPATIBLE
- Top failure categories:
  - MISSING_PLAN_TOOL_CALL: 1

### gpt-oss:20b

- Digest: `17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7`
- Artifact size: 12.8 GiB
- Parameter size: 20.9B
- Quantization: MXFP4
- Declared capabilities: completion, thinking, tools
- Observed loaded size: n/a
- Observed VRAM allocation: n/a
- Observed offload mode: UNKNOWN (runtime metadata unavailable)

#### Protocol: structured_output
- Qualification: NOT_QUALIFIED
- Top failure categories:
  - DRAFT_COMPILATION_FAILED: 7
  - CANONICAL_VALIDATION_FAILED: 5
  - WRONG_ACTION_TYPE: 5
  - WRONG_WORKSPACE_REFERENCE: 3
- Failed case IDs: planner.active_workspace_source_add, planner.ambiguous_missing_workspace_target, planner.attachment_ingestion, planner.mixed_source_ordinal_routing, planner.source_candidate_attach_ordinal, planner.source_candidate_list, planner.source_list_named_workspace

#### Protocol: single_plan_tool
- Qualification: SCHEMA_INCOMPATIBLE
- Top failure categories:
  - DRAFT_VALIDATION_FAILED: 1

### mistral-small3.2:24b

- Digest: `5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b`
- Artifact size: 14.1 GiB
- Parameter size: n/a
- Quantization: n/a
- Declared capabilities: n/a
- Observed loaded size: n/a
- Observed VRAM allocation: n/a
- Observed offload mode: UNKNOWN (runtime metadata unavailable)

#### Protocol: structured_output
- Qualification: PROVIDER_ERROR
- Top failure categories:
  - PROVIDER_ERROR: 1

#### Protocol: single_plan_tool
- Qualification: PROTOCOL_UNSUPPORTED
- Top failure categories:
  - PROTOCOL_UNSUPPORTED: 1

## 12. Reproduction

```powershell
uv run python applications/local_workspace_application/scripts/run-local-model-qualification.py
```

- Generated at (UTC): 2026-07-29T12:07:20.815956+00:00
- Commit: 5fe424b13c92d013c4c52de55b8024ab745d9a39
- Configuration SHA-256: `b167e7007ebdcd5cc27c04dae3ddd082bcb7a54871e76b1b7c8262fd71a07a67`

## 13. Limitations

- This benchmark measures LKW conversational planning semantics only.
- Results are tied to observed hardware and installed model digests when available.
- `single_plan_tool` is experimental and not used in production.
- No universal minimum hardware requirements are implied.
