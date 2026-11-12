# ADR-PDC-05 — Minimal typed evaluation contracts, reproducible runs and advisory comparison

| Field | Value |
| --- | --- |
| **Document ID** | ADR-PDC-05 |
| **Status** | **Proposed** (pending OD-PDC-ADR-01) |
| **Campaign** | PDC-X P3 (T-IGX-036-D) |
| **Domain** | PDC-D5 Evaluation |
| **Requirements** | PDC-D5-R001..R007, R009..R014, R017, R018 — see [`PDC_DOMAIN_REQUIREMENTS_MATRIX.md`](../qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md) |
| **FRZ** | FRZ-CTR-01, FRZ-TYP-01, FRZ-TRC-04, FRZ-TRC-05, FRZ-EVOL-06, FRZ-GOV-01, FRZ-GOV-02, FRZ-TEN-07 |

## Context

- `EvalResult` has no run, tenant, model, dataset version or code SHA; `GoldenDatasetAsset` has a free-string version and no digest; the only comparison is a pass-rate delta; promotion gates G0–G2 only check that files exist; no safety evaluator exists.
- Canonical decision authority is `DECISION_SYSTEM.md` (CRITIC_VERIFICATION is legacy; PDC-P1-F04).

## Decision (proposed)

- Add typed contracts in `intergrax/contracts/` — `EvalDataset` (content digest, version), `EvalScenario`, `EvalRun` (agent/model/prompt/profile revisions, code SHA, dataset digest, seed, tenant) and `EvalComparison` (baseline vs candidate, per-metric deltas, cost-to-success, latency).
- One evaluator contract with explicit kinds: `DETERMINISTIC`, `MODEL_ASSISTED` (isolated judge, never mixed with rule results), `HUMAN`, `SAFETY` (implemented through existing guardrail integrations).
- Runs and comparisons are persisted as canonical evidence (`EvalEvidenceReport` path); file-based registries get explicit concurrency semantics or are replaced by the evidence store.
- Evaluation is **advisory**: comparison results are inputs to governance-owned promotion; an evaluation PASS can never authorize deployment by itself.
- Customer-specific datasets and production benchmarks are out of scope.

## Alternatives rejected

- Folding evaluation into ENT-AUDIT-X (that certifies architecture, not behavior).
- A promotion gate owned by the evaluation package (Governance ≠ Evaluation).

## Verification

`tests/qualification/pdc_x/d5/` plus `xd_08` (digest-identical rerun; eval PASS cannot authorize promotion).

## Status note

Proposed only. No implementation before OD-PDC-ADR-01 accepts this ADR; product changes then follow OD-PDC-SEQ-01 (merge after MCP-SERVER-X P8).
