# TOKEN-10H / TOKEN-10I proof evidence

## TOKEN-10H (Qwen 7B AWQ)

- Latest live proof status: `GENUINE_LIVE_MODEL_BEHAVIOR_MISMATCH`
- Provider/model: `local vLLM` / `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Runs: `2`
- Offline: `PASS`
- Runtime safety: `PASS`
- Model behavioral: `PARTIAL` (`14/16`)
- Public promotion: `WITHHELD`
- Qualification process: `CLOSED` (`NOT QUALIFIED`)

Latest checked-in Qwen 7B evidence is the post-SAFETY-1 frozen qualification
rerun. Historical pre-SAFETY-1 `046c3ad6` evidence remains the truthful prior
negative proof (runtime safety was FAIL then). Model mismatches are recorded
separately from deterministic runtime fail-safe success. Raw content, protected
values, secrets, paths, and provider cache claims are not published.

Artifacts:

- [Live run 1](token-10h-live-run-1.safe.json)
- [Live run 2](token-10h-live-run-2.safe.json)
- [Evaluation and claim matrix](token-10h-evaluation.safe.json)
- [Closeout report](token-10h-live-report.md)

## TOKEN-10I (Qwen3 14B AWQ)

- Status: `BLOCKED_HARDWARE_CAPACITY_FINAL`
- Candidate: `Qwen/Qwen3-14B-AWQ`
- Attempt A: default `max_model_len` / `gpu-memory-utilization 0.85` → READY FAIL
  (weight load PASS `9.44 GiB`, KV available `-0.89 GiB`)
- Attempt B: controlled `max_model_len 8192` / `gpu-memory-utilization 0.95`,
  no offload → READY FAIL (CUDA free-memory gate `10.79/11.99 GiB` < `11.39 GiB`)
- Frozen qualification: not started
- Evidence: [Feasibility report](token-10i-qwen3-14b-awq-feasibility.md)
