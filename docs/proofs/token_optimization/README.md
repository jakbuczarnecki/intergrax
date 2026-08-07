# TOKEN-10H proof evidence

- Latest live proof status: `GENUINE_LIVE_MODEL_BEHAVIOR_MISMATCH`
- Provider/model: `local vLLM` / `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Runs: `2`
- Offline: `PASS`
- Runtime safety: `FAIL`
- Model behavioral: `PARTIAL` (`14/16`)
- Public promotion: `WITHHELD`

The checked-in evidence is a safe, auditable negative live proof after correct
7B AWQ model load. It records model mismatches separately from infrastructure
failures and does not publish raw content, protected values, secrets, paths, or
provider cache claims.

Artifacts:

- [Live run 1](token-10h-live-run-1.safe.json)
- [Live run 2](token-10h-live-run-2.safe.json)
- [Evaluation and claim matrix](token-10h-evaluation.safe.json)
- [Closeout report](token-10h-live-report.md)
