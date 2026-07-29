<!--
GENERATED FILE.
Do not edit benchmark results manually.
Regenerate with:
uv run python applications/local_workspace_application/scripts/run-local-model-qualification.py
-->

# LKW Local Ollama Model Qualification

## 1. Scope and interpretation

This is an LKW-specific benchmark for conversational interaction planning. It is not a universal LLM ranking. Results apply to the exact model digests and Ollama version shown below.

## 2. Executive summary

No tested model/protocol pair met the full LKW qualification threshold.

## 3. Recommended configuration

No tested model/protocol pair met the full LKW qualification threshold.

## 4. Benchmark methodology

- Corpus version: `lkw.local_model_qualification.corpus.v1`
- Production semantic prompt via `build_planning_messages()`
- Structured output remains the current production transport
- `single_plan_tool` is a benchmark candidate only
- Submission tool `submit_conversation_interaction_draft` does not execute operations
- Repair is disabled in benchmark scoring (`repair_attempts=0`)
- Tool protocol adds only this transport instruction after the production system message:
  Call submit_conversation_interaction_draft exactly once with the complete semantic draft as its arguments. This submission tool does not execute any operation. Do not answer in plain text and do not call any other tool.

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
- Version: 0.32.1

## 7. Tested model inventory

| Model | Role | Installed | Digest |
| --- | --- | --- | --- |
| qwen2.5:14b | baseline | True | `n/a` |
| qwen3:14b | same-size-generation-candidate | True | `n/a` |
| llama3.1:8b | lightweight-control | True | `n/a` |
| gpt-oss:20b | agentic-candidate | False | `n/a` |
| mistral-small3.2:24b | instruction-following-candidate | False | `n/a` |

## 8. Model × protocol comparison

| Model | Role | Protocol | Capabilities | Schema probe | Samples | Semantic success | Invalid drafts | Provider failures | Unsafe state changes | Median latency | p95 latency | Execution mode | Qualification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:14b | baseline | structured_output | completion, tools | PASS | 36 | 44.4% | 0 | 0 | 5 | 4086 | 10265 | UNKNOWN | NOT_QUALIFIED |
| qwen2.5:14b | baseline | single_plan_tool | completion, tools | PROVIDER_ERROR | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | UNKNOWN | PROVIDER_ERROR |
| qwen3:14b | same-size-generation-candidate | structured_output | completion, thinking, tools | PASS | 36 | 52.8% | 0 | 0 | 2 | 17829 | 96758 | UNKNOWN | NOT_QUALIFIED |
| qwen3:14b | same-size-generation-candidate | single_plan_tool | completion, thinking, tools | PROVIDER_ERROR | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | UNKNOWN | PROVIDER_ERROR |
| llama3.1:8b | lightweight-control | structured_output | n/a | PASS | 36 | 36.1% | 0 | 0 | 9 | 3352 | 12527 | UNKNOWN | NOT_QUALIFIED |
| llama3.1:8b | lightweight-control | single_plan_tool | n/a | PROVIDER_ERROR | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | UNKNOWN | PROVIDER_ERROR |

## 9. Safety and state-change results

- qwen2.5:14b / structured_output: unsafe state changes = 5
- qwen2.5:14b / single_plan_tool: unsafe state changes = 0
- qwen3:14b / structured_output: unsafe state changes = 2
- qwen3:14b / single_plan_tool: unsafe state changes = 0
- llama3.1:8b / structured_output: unsafe state changes = 9
- llama3.1:8b / single_plan_tool: unsafe state changes = 0

## 10. Failure categories

### qwen2.5:14b / structured_output
- CANONICAL_VALIDATION_FAILED: 6
- DRAFT_COMPILATION_FAILED: 7
- UNNECESSARY_WORKSPACE_ACTIVATE: 5
- WRONG_ACTION_COUNT: 5
- WRONG_ACTION_TYPE: 2
- WRONG_WORKSPACE_REFERENCE: 1

### qwen3:14b / structured_output
- CANONICAL_VALIDATION_FAILED: 5
- DRAFT_COMPILATION_FAILED: 9
- UNNECESSARY_CLARIFICATION: 1
- UNNECESSARY_WORKSPACE_ACTIVATE: 2
- WRONG_ACTION_COUNT: 1
- WRONG_ACTION_TYPE: 2
- WRONG_WORKSPACE_REFERENCE: 1

### llama3.1:8b / structured_output
- DRAFT_COMPILATION_FAILED: 7
- MISSING_CLARIFICATION: 3
- UNEXPECTED_STATE_CHANGE: 4
- UNNECESSARY_CLARIFICATION: 1
- UNNECESSARY_WORKSPACE_ACTIVATE: 5
- WRONG_ACTION_COUNT: 7
- WRONG_ACTION_TYPE: 5
- WRONG_SOURCE_EXTRACTION: 1
- WRONG_WORKSPACE_REFERENCE: 7


## 11. Per-model details

### qwen2.5:14b

- Digest: `n/a`
- Artifact size: n/a
- Parameter size: 14.8B
- Quantization: Q4_K_M
- Declared capabilities: completion, tools
- Observed VRAM allocation: n/a
- Observed offload mode: UNKNOWN

#### Protocol: structured_output
- Qualification: NOT_QUALIFIED
- Top failure categories:
  - DRAFT_COMPILATION_FAILED: 7
  - CANONICAL_VALIDATION_FAILED: 6
  - UNNECESSARY_WORKSPACE_ACTIVATE: 5
  - WRONG_ACTION_COUNT: 5
  - WRONG_ACTION_TYPE: 2
- Failed case IDs: planner.active_workspace_source_add, planner.ambiguous_missing_workspace_target, planner.attachment_ingestion, planner.mixed_source_ordinal_routing, planner.source_candidate_attach_ordinal, planner.source_candidate_list, planner.source_list_named_workspace, planner.target_workspace_without_activation

#### Protocol: single_plan_tool
- Qualification: PROVIDER_ERROR

### qwen3:14b

- Digest: `n/a`
- Artifact size: n/a
- Parameter size: 14.8B
- Quantization: Q4_K_M
- Declared capabilities: completion, thinking, tools
- Observed VRAM allocation: n/a
- Observed offload mode: UNKNOWN

#### Protocol: structured_output
- Qualification: NOT_QUALIFIED
- Top failure categories:
  - DRAFT_COMPILATION_FAILED: 9
  - CANONICAL_VALIDATION_FAILED: 5
  - UNNECESSARY_WORKSPACE_ACTIVATE: 2
  - WRONG_ACTION_TYPE: 2
  - UNNECESSARY_CLARIFICATION: 1
- Failed case IDs: planner.active_workspace_source_add, planner.ambiguous_missing_workspace_target, planner.attachment_ingestion, planner.mixed_source_ordinal_routing, planner.source_candidate_attach_ordinal, planner.source_candidate_list, planner.source_list_named_workspace

#### Protocol: single_plan_tool
- Qualification: PROVIDER_ERROR

### llama3.1:8b

- Digest: `n/a`
- Artifact size: n/a
- Parameter size: 8.0B
- Quantization: Q4_K_M
- Declared capabilities: n/a
- Observed VRAM allocation: n/a
- Observed offload mode: UNKNOWN

#### Protocol: structured_output
- Qualification: NOT_QUALIFIED
- Top failure categories:
  - DRAFT_COMPILATION_FAILED: 7
  - WRONG_ACTION_COUNT: 7
  - WRONG_WORKSPACE_REFERENCE: 7
  - UNNECESSARY_WORKSPACE_ACTIVATE: 5
  - WRONG_ACTION_TYPE: 5
- Failed case IDs: planner.active_workspace_source_add, planner.ambiguous_missing_workspace_target, planner.attachment_ingestion, planner.explicit_workspace_activation, planner.explicit_workspace_delete, planner.mixed_source_ordinal_routing, planner.source_candidate_attach_ordinal, planner.source_candidate_list, planner.source_list_named_workspace, planner.target_workspace_without_activation, planner.url_question_not_ingestion

#### Protocol: single_plan_tool
- Qualification: PROVIDER_ERROR

### gpt-oss:20b

- Digest: `n/a`
- Artifact size: n/a
- Parameter size: n/a
- Quantization: n/a
- Declared capabilities: n/a
- Observed VRAM allocation: n/a
- Observed offload mode: UNKNOWN

### mistral-small3.2:24b

- Digest: `n/a`
- Artifact size: n/a
- Parameter size: n/a
- Quantization: n/a
- Declared capabilities: n/a
- Observed VRAM allocation: n/a
- Observed offload mode: UNKNOWN

## 12. Reproduction

```powershell
uv run python applications/local_workspace_application/scripts/run-local-model-qualification.py
```

- Generated at (UTC): 2026-07-29T09:04:31.503458+00:00
- Commit: fe2b01270d360c61324c5937acc35aa1b5ca9226
- Configuration SHA-256: `16daa87a3f01f0d919323a58877ee029899aabb436859a1d4efadb8b286de082`

## 13. Limitations

- This benchmark measures LKW conversational planning semantics only.
- Results are tied to observed hardware and installed model digests.
- `single_plan_tool` is experimental and not used in production.
- No universal minimum hardware requirements are implied.
