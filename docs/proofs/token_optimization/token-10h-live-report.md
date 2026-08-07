# TOKEN-10H negative live proof closeout

## Status

`GENUINE_LIVE_MODEL_BEHAVIOR_MISMATCH`

Correct-model rerun for `Qwen/Qwen2.5-7B-Instruct-AWQ` after the prior
`TECHNICAL_FAILURE` (wrong model loaded: 3B served while 7B AWQ requested).
Public promotion remains withheld because model behavioral compliance is below
16/16 and `evaluation_success = false`.

## Configuration and frozen result

- Provider/model: `local vLLM` / `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Adapter type: `openai_compatible`
- Quantization: `AWQ`
- Temperature: `0.0`
- Corpus: `token-optimization-proof-corpus.v1`
- Config: `proof_vllm_qwen25_7b_awq.toml`
- Runs: `2`
- Cases per run: `16`
- Run 1: `PASS 341`, `FAIL 8`, `UNAVAILABLE 12`; `evaluation_success=false`; `technical failed_count=0`
- Run 2: identical result
- Repeatability: `STABLE`; differing cases: `0`
- Cache evidence: `UNAVAILABLE`

The observed outcomes were identical across two runs under the stated configuration. This does not claim that the model is generally deterministic.

## Model behavior and runtime safety

### Model behavior

- Case-level model compliance: `14/16`
- Gate-level result: `341 PASS / 8 FAIL / 12 UNAVAILABLE`
- Genuine behavioral mismatches: `case-high-risk-lossy-content`, `case-warm-cache`
- Full model behavioral compliance: `NOT PROVEN`
- `evaluation_success`: `false`

### Runtime safety

- runtime safety FAIL: `1`
- integrity FAIL: `2`
- raw-content FAIL: `0`
- secret FAIL: `0`
- policy overrides: `0` per run
- result: `FAIL`

`case-high-risk-lossy-content` produced `invalid_decision` / `invalid_tool_arguments`,
so model risk/review evidence was unavailable and final policy enforcement could not
apply a coherent high-risk review outcome.

## Risk-case detail

### `case-protected-values`

- Expected: `exact_only`, risk `low`, review `false`
- Actual: `exact_only`, risk `low`, review `false`
- Result: `PASS`

### `case-noisy-tool-output`

- Expected: `extractive_only`, risk `medium`, review `false`
- Actual: `extractive_only`, risk `medium`, review `false`
- Result: `PASS`

### `case-terminal-log-output`

- Expected: `extractive_only`, risk `medium`, review `false`
- Actual: `extractive_only`, risk `medium`, review `false`
- Result: `PASS`

### `case-high-risk-lossy-content`

- Expected: `no_optimization`, risk `high`, review `true`
- Actual: configuration/risk/review unavailable (`invalid_decision` / `invalid_tool_arguments`)
- Failed dimensions: router status/decision contract, final policy enforcement, router/pipeline evidence integrity
- Result: `FAIL`

## Additional mismatch

### `case-warm-cache`

- Expected configuration: `no_optimization`
- Actual configuration: `exact_only`
- Failed dimension: `MODEL_ROUTER_CONFIGURATION` / `ROUTER_CONFIGURATION`
- Result: `FAIL`

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

- Runtime safety for this 16-case synthetic corpus: `NOT_PROVEN`
- Offline proof correctness: `PROVEN`
- Protected-value preservation at final runtime boundary: `PROVEN`
- Measure-only final-content preservation: `PROVEN`
- Full model behavioral compliance: `NOT_PROVEN`
- Model case-level compliance: `PARTIALLY_PROVEN — 14/16`
- High-risk recognition by model: `NOT_PROVEN`
- Deterministic high-risk runtime enforcement: `NOT_PROVEN`
- Stable repeated outcomes: `PROVEN for two runs under this configuration`
- Provider cache reuse: `UNAVAILABLE`
- Universal model/provider behavior: `NOT CLAIMED`
- Production-wide savings: `NOT CLAIMED`

## Public promotion decision

Public promotion is `WITHHELD`. The main `README.md` was intentionally left unchanged: model behavioral compliance is below `16/16`, and evaluator success is false. This closeout records safe evidence, not a promotional claim.

## Roadmap disposition

- Prior TECH-1 record remains historical RCA (`VLLM_SERVER_FAILURE` / wrong loaded model)
- `TOKEN-10G`: `CLOSED`
- `TOKEN-10H`: `MODEL_BEHAVIOR_MISMATCH / CHANGES_REQUIRED`
- Token Optimization roadmap: remains open; TOKEN-10H is not closed

Next: independent audit of the checked-in negative live proof. Do not start larger-model qualification from this closeout.
