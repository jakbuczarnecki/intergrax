# TOKEN-10I Qwen3 14B AWQ feasibility

## Status

`BLOCKED_HARDWARE_CAPACITY`

Frozen qualification for `Qwen/Qwen3-14B-AWQ` did not start. Phases B–E were
not executed.

## Candidate and runtime

- Exact model: `Qwen/Qwen3-14B-AWQ`
- Quantization: official AWQ (runtime path `awq_marlin`)
- vLLM image/version: `vllm/vllm-openai:v0.23.0`
- Local selection: `VLLM_MODEL=Qwen/Qwen3-14B-AWQ`
- Compose first attempt retained `--tool-call-parser hermes`
- No memory/context/KV tuning was performed in this task

## Hardware evidence

- GPU: NVIDIA GeForce RTX 4080 Laptop GPU
- Total VRAM: `12282 MiB`
- Model architecture: `Qwen3ForCausalLM`
- vLLM selected `max_model_len`: `40960`
- Model weight memory: `9.44 GiB` (load PASS)
- Available KV cache memory after load: `-0.89 GiB`
- Existing setting: `--gpu-memory-utilization 0.85`
- Failure stage: KV-cache initialization after weight load
- Error class: `ValueError: No available memory for the cache blocks`
- Model READY: FAIL

## Gates not reached

- `/v1/models` identity
- OpenAI chat smoke
- `tool_choice="required"` tool smoke
- Token Optimization `short-clean` / high-risk smoke
- Semantic prequalification
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
- Public README/marketing: not modified

## Comparison note

Qwen 7B reference remains `14/16` `MODEL_BEHAVIOR_MISMATCH` with runtime safety
PASS. No Qwen3 frozen-contract compliance delta exists because the model did
not become READY under the canonical local setup.
