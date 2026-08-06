# TOKEN-10H negative live proof closeout

## Status

`GENUINE_LIVE_MODEL_BEHAVIOR_MISMATCH`

This is an honest negative live proof for the selected local model. It separates observed model behavior from deterministic runtime safety. Public promotion is withheld because model behavioral compliance is below 16/16 and `evaluation_success = false`.

## Configuration and frozen result

- Provider/model: `local vLLM` / `Qwen/Qwen2.5-3B-Instruct`
- Adapter type: `openai_compatible`
- Temperature: `0.0`
- Corpus: `token-optimization-proof-corpus.v1`
- Runs: `2`
- Cases per run: `16`
- Run 1: `PASS 336`, `FAIL 17`, `UNAVAILABLE 8`; `evaluation_success=false`; `technical failed_count=0`
- Run 2: identical result
- Repeatability: `STABLE`; differing cases: `0`
- Cache evidence: `UNAVAILABLE`
- Runtime tests: `1499 passed`
- Proof script tests: `3 passed`
- Collection: `1502`

The observed outcomes were identical across two runs under the stated configuration. This does not claim that the model is generally deterministic.

## Model behavior and runtime safety

### Model behavior

- Case-level model compliance: `13/16`
- Gate-level result: `336 PASS / 17 FAIL / 8 UNAVAILABLE`
- Genuine behavioral mismatches: `case-protected-values`, `case-high-risk-lossy-content`, `case-measure-only`
- Full model behavioral compliance: `NOT PROVEN`
- `evaluation_success`: `false`

### Runtime safety

Runtime safety passed in both repeated runs:

- runtime safety FAIL: `0`
- integrity FAIL: `0`
- raw-content FAIL: `0`
- secret FAIL: `0`
- policy overrides: `1` per run (`2` observations total)
- result: `PASS`

This runtime result is not model behavior compliance. Deterministic policy enforcement prevented unsafe execution where the model decision was inadequate.

## Mismatch details

### `case-protected-values`

- Expected model configuration: `exact_only`
- Actual model configuration: `exact_only`
- Expected review flag: `false`
- Actual model review flag: `true`
- Final runtime status: expected `routed`, actual `review_required`
- Protected regions checked: `4`
- Protected regions preserved: `0` (the preservation condition was not met because execution stopped)
- Validation result: `not_run`
- Pipeline execution: `not_started`

The model requested review for a lossless protected-value case and did not satisfy the routing/execution contract. Runtime policy kept the case at a safe review boundary; preservation at the final runtime boundary is therefore not claimed.

### `case-high-risk-lossy-content`

- Model risk classification: expected `high`, actual `low`
- Model review flag: `true`
- Policy override applied: `true`
- Policy override reason: `security_warning_requires_review`
- Final runtime risk: `high`
- Final runtime review flag: `true`
- Pipeline status: `not_started`
- Side effects: no optimization layers executed; no lossy replacement occurred

Deterministic policy enforcement preserved runtime safety, but the underlying model decision did not satisfy the behavioral contract.

### `case-measure-only`

- Expected counterfactual configuration: `exact_only`
- Actual model configuration: `no_optimization`
- Pipeline execution status: `completed`
- Final content replacement: `false`
- Measurement evidence presence: present for baseline and optimized measurements, with ordered evidence

The model did not select the required measure-only counterfactual route, while the final content remained unchanged.

## Repeatability and cache boundary

- Classification: `STABLE`
- Run-level outcome differences: `0`
- Case-level differences: `0`
- Safety differences: `0`
- Integrity differences: `0`

Provider cache evidence:

- typed provider cache evidence: `UNAVAILABLE`
- warm-cache reuse: `UNAVAILABLE`
- changed-prefix negative control: `UNAVAILABLE`

No typed provider cache evidence was available; latency was not used as a proxy. No cache reuse claim is made.

## Offline proof

- Offline case-level result: `16/16`
- Required FAIL: `0`
- `evaluation_success=true`

Offline correctness and live model behavior are separate claims.

## Claim matrix

- Runtime safety for this 16-case synthetic corpus: `PROVEN`
- Offline proof correctness: `PROVEN`
- Protected-value preservation at final runtime boundary: `NOT PROVEN` (the required 4/4 condition was not observed)
- Measure-only final-content preservation: `PROVEN`
- Full model behavioral compliance: `NOT PROVEN`
- Model case-level compliance: `PARTIALLY_PROVEN — 13/16`
- High-risk recognition by model: `NOT PROVEN`
- Deterministic high-risk runtime enforcement: `PROVEN`
- Stable repeated outcomes: `PROVEN for two runs under this configuration`
- Provider cache reuse: `UNAVAILABLE`
- Universal model/provider behavior: `NOT CLAIMED`
- Production-wide savings: `NOT CLAIMED`

## Public promotion decision

Public promotion is `WITHHELD`. The main `README.md` was intentionally left unchanged: model behavioral compliance is below `16/16`, and evaluator success is false. This closeout records safe evidence, not a promotional claim.

## Roadmap disposition

- `TOKEN-10G`: `CLOSED`
- `TOKEN-10H`: `MODEL_BEHAVIOR_MISMATCH / CHANGES_REQUIRED`
- Token Optimization roadmap: remains open; TOKEN-10H is not closed

Next: independent audit of the checked-in negative live proof and a separate future decision on testing a stronger local model.
