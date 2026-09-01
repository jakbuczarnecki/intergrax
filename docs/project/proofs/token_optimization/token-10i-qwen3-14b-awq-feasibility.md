# TOKEN-10I Qwen3 14B AWQ feasibility

## Status

`BLOCKED_HARDWARE_CAPACITY_FINAL`

Frozen qualification for `Qwen/Qwen3-14B-AWQ` did not start. Phases B–E were
not executed. The model is operationally unsuitable for this GPU under the
approved no-offload capacity profile.

## Candidate and runtime

- Exact model: `Qwen/Qwen3-14B-AWQ`
- Quantization: official AWQ (runtime path `awq_marlin`)
- vLLM image/version: `vllm/vllm-openai:v0.23.0`
- Local selection: `VLLM_MODEL=Qwen/Qwen3-14B-AWQ`
- Tool parser retained `hermes` after installed vLLM 0.23.0 + model chat-template
  check (Hermes-style `<tool_call>` JSON; not Qwen3-Coder `qwen3_xml`)
- No CPU offload, swap, KV offload, FP8 KV, alternate quantization, TP, or
  speculative decoding

## Attempt A - canonical default context / 0.85 (preserved)

- Profile: default `max_model_len` selected by vLLM / `--gpu-memory-utilization 0.85`
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU, total VRAM `12282 MiB`
- Model architecture: `Qwen3ForCausalLM`
- vLLM selected `max_model_len`: `40960`
- Model weight memory: `9.44 GiB` (load PASS)
- Available KV cache memory after load: `-0.89 GiB`
- Failure stage: KV-cache initialization after weight load
- Error class: `ValueError: No available memory for the cache blocks`
- Model READY: FAIL
- Result recorded as `BLOCKED_HARDWARE_CAPACITY` (not final)

## Attempt B - controlled capacity profile 8192 / 0.95 (this task)

- Profile: `--max-model-len 8192` / `--gpu-memory-utilization 0.95`
- Applied via ephemeral local compose override only (tracked production compose
  unchanged; no permanent production defaults edited)
- Actual args observed: `max_model_len=8192`, `gpu_memory_utilization=0.95`,
  `tool_call_parser=hermes`, AWQ path retained
- Host idle VRAM evidence before start: ~`52–64 MiB` used / ~`11934–11946 MiB`
  free of `12282 MiB`
- Engine CUDA free-memory gate at startup: free `10.79/11.99 GiB` less than
  desired utilization budget `0.95` (`11.39 GiB`)
- Failure stage: device memory reservation before weight load
- Error class: `ValueError: Free memory on device cuda:0 (...) is less than
  desired GPU memory utilization (0.95, ...)`
- Model weight memory: not reached under this profile
- Available KV cache memory / KV token capacity: not reached
- CUDA/OOM during weight/KV init: not reached (failed earlier gate)
- Model READY: FAIL
- No further tuning permitted by task scope

## Gates not reached

- `/v1/models` identity
- OpenAI chat smoke
- `tool_choice="required"` tool smoke
- Token Optimization `short-clean` / high-risk smoke
- Semantic prequalification
- Context-headroom measurement across 16 canonical cases
- Two frozen qualification runs

## Integrity

- Prompt delta: none
- Corpus delta: none
- Risk semantics delta: none
- Threshold delta: none
- Evaluator semantic delta: none
- Runtime safety semantic delta: none
- Post-result tuning: none
- Historical Qwen 7B TOKEN-10H evidence: preserved
- Attempt A 40k/0.85 evidence: preserved as separate first failure
- Public README/marketing: not modified
- Tracked vLLM compose defaults: not modified

## Comparison note

Qwen 7B reference remains `14/16` `MODEL_BEHAVIOR_MISMATCH` with runtime safety
PASS. No Qwen3 frozen-contract compliance delta exists because the model did
not become READY under either recorded capacity profile.
