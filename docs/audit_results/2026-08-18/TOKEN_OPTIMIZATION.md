# TOKEN_OPTIMIZATION — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer/capability:** TOKEN_OPTIMIZATION
- **Tier(s):** Tier-0 `intergrax/runtime/token_optimization/` — pipeline runner, protected regions, receipts, LLM router, cache-aware runtime/orchestration, emission helpers
- **audited_sha:** `061e03f6dc6160d8f857fbda29d1d6848d040a8d`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md`
- **Plan doc(s):**
  - `docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md`
- **Scope in:**
  - `TokenOptimizationPipelineRunner` MEASURE_ONLY aggregation and measurement semantics
  - protected-region detection and `validate_protected_regions()`
  - `TokenOptimizationPolicy`, `TokenOptimizationLayerDescriptor`, pipeline validation gates
  - `OutputPolicy` vs canonical pipeline safety authority
  - receipt/observability emission lifecycle (`emit_receipts`, `emit_observability`, `maybe_emit_token_optimization_outcome()`)
  - `CompressionReceipt` / `_derive_receipt_id()` identity
  - architecture/plan TOKEN-10G/H/I lifecycle current-state parity
  - historical TOKEN-1..9, TOKEN-10E/F/F-EVIDENCE-EXTENSION, TOKEN-10G/H/I delivery as positive controls
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - second tokenizer or private Token Optimization telemetry bus
  - rewriting historical TOKEN closeout rows or converting TOKEN-10H NOT QUALIFIED into success
  - creating domain-layer `docs/project/architecture/TOKEN_OPTIMIZATION.md` or `docs/project/maintainers/plans/TOKEN_OPTIMIZATION.md`
- **Prior audit reference(s):** [`LLM_ADAPTERS`](LLM_ADAPTERS.md); [`OBSERVABILITY_EVIDENCE`](OBSERVABILITY_EVIDENCE.md); [`CONTEXT_ENGINEERING`](CONTEXT_ENGINEERING.md)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** —

## Executive summary

**Verdict: FAIL.** Four accepted HIGH and two accepted MEDIUM findings show that MEASURE_ONLY aggregation labels character counts as measured tokens; protected-region validation checks substring existence only; lossy layers can run with validation fully disabled when policy and descriptor both waive it; effective observability policy is disconnected from the separate default-off emission helper; receipt identity collapses across tenants/runs; and architecture current-state drifted from plan on TOKEN-10G/H/I. Positive controls: Token Optimization is a cross-layer platform capability with LKW as later proof/client; no second tokenizer is architecturally intended; pipeline rollback, router catalog discipline, policy OFF blocks, cache-aware safe reporting, and metadata sanitization remain intact; TOKEN-10H stays NOT QUALIFIED and TOKEN-10I hardware-blocked. Remediation is **PLANNED**, not implemented. Findings harden the existing Token Optimization capability — no second engine required.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-TOKEN_OPTIMIZATION-01

- **Severity:** HIGH
- **Category:** MEASUREMENT INTEGRITY / FALSE TOKEN ACCOUNTING
- **Status at publication:** ACCEPTED
- **Remediation block:** TOKEN-MEASUREMENT-EVIDENCE-INTEGRITY
- **Claim falsified:** Only canonical tokenizer/provider token measurements use the TOKENS unit and MEASURED confidence; character approximation is a different unit.
- **Observation:** `TokenOptimizationPipelineRunner` MEASURE_ONLY aggregation calls `_measure_only_aggregate_measurement()`. That function assigns `baseline_tokens = len(original_content)` and `optimized_tokens = len(optimized_content)` — character counts — then constructs `TokenSavingsMeasurement` with `confidence=MEASURED` and only records `metadata={"measurement_basis": "character_count"}`. Architecture explicitly says baseline token counts should use the existing LLM-adapter/tokenizer path and Token Optimization must not create a second token counting semantics.
- **Location:**
  - `intergrax/runtime/token_optimization/pipeline.py` — `_measure_only_aggregate_measurement()`
  - `intergrax/runtime/token_optimization/contracts.py` — `TokenSavingsMeasurement`
- **Reproduction:** Run MEASURE_ONLY pipeline path; inspect measurement object — character lengths labeled MEASURED tokens.
- **Impact:** Proof, cost calculation, promotion gates, and savings percentages may treat characters as measured tokens; cross-link [`LLM_ADAPTERS`](../../project/architecture/LLM_ADAPTERS.md).
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOKEN_OPTIMIZATION-02

- **Severity:** HIGH
- **Category:** SAFETY / PROTECTED CONTENT INTEGRITY
- **Status at publication:** ACCEPTED
- **Remediation block:** TOKEN-PROTECTED-TRANSFORMATION-INTEGRITY
- **Claim falsified:** Protected-region validation preserves exact protected-region occurrences, not substring existence alone.
- **Observation:** ProtectedRegion detection records value and original start/end occurrence. `validate_protected_regions()` validates each original region only by `region.value in optimized_content`. It does not enforce one-to-one occurrence preservation, multiplicity, ordering, or source-specific structural relation. Multiple identical protected occurrences can therefore all pass validation when only one copy survives optimization.
- **Location:**
  - `intergrax/runtime/token_optimization/protected_regions.py` — detection and `validate_protected_regions()`
- **Reproduction:** Optimize content with duplicate protected occurrences; remove all but one copy; observe validation pass.
- **Impact:** Lossy optimization can silently drop protected duplicates; architecture describes exact protected-region preservation.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOKEN_OPTIMIZATION-03

- **Severity:** HIGH
- **Category:** EXTENSIBILITY / SAFETY BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** TOKEN-PROTECTED-TRANSFORMATION-INTEGRITY
- **Claim falsified:** LOSSY transformation always requires the platform minimum validation contract; invalid policy states fail closed.
- **Observation:** `TokenOptimizationPolicy` permits `allow_lossy=True` and `require_validation=False` without contract failure. `TokenOptimizationLayerDescriptor` permits `safety_class=LOSSY` and `requires_validation=False` without contract failure. Pipeline only performs protected-region validation when `policy.require_validation OR descriptor.requires_validation`. Therefore a registered lossy layer/plugin can run with validation completely disabled when both inputs are false. `OutputPolicy` has a separate validator that rejects `allow_lossy + !require_validation`, but the canonical optimization pipeline does not consume that validation as its safety authority.
- **Location:**
  - `intergrax/runtime/token_optimization/contracts.py` — `TokenOptimizationPolicy`, `TokenOptimizationLayerDescriptor`
  - `intergrax/runtime/token_optimization/pipeline.py` — validation gate
  - `intergrax/runtime/token_optimization/output_policy.py` — separate validator
- **Reproduction:** Register lossy layer with both policy and descriptor validation disabled; run pipeline; observe lossy transform without protected-region validation.
- **Impact:** Plugin self-declaration can bypass canonical minimum safety.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOKEN_OPTIMIZATION-04

- **Severity:** HIGH
- **Category:** OBSERVABILITY / PAPER POLICY
- **Status at publication:** ACCEPTED
- **Remediation block:** TOKEN-MEASUREMENT-EVIDENCE-INTEGRITY
- **Claim falsified:** One effective Token Optimization policy resolves transformation behavior and receipt/observability obligations.
- **Observation:** `TokenOptimizationPolicy` contains `emit_receipts=True` and `emit_observability=True` and architecture describes receipt + observability emission as part of the optimization lifecycle. The pipeline itself returns result/receipt metadata but does not emit. Actual emission is exposed through separate helper `maybe_emit_token_optimization_outcome()` using an independent `TokenOptimizationEmissionPolicy` whose default is `enabled=False`. No canonical binding was found from effective `TokenOptimizationPolicy.emit_observability` to the separate emission policy. Repository usage of `maybe_emit_token_optimization_outcome()` is helper/tests, not the canonical pipeline/runtime execution path.
- **Location:**
  - `intergrax/runtime/token_optimization/contracts.py` — `TokenOptimizationPolicy`
  - `intergrax/runtime/token_optimization/emission.py` — `maybe_emit_token_optimization_outcome()`, `TokenOptimizationEmissionPolicy`
  - `intergrax/runtime/token_optimization/pipeline.py` — outcome without canonical emission
- **Reproduction:** Configure policy with observability enabled; run canonical pipeline path without calling emission helper; observe no HOS emission despite policy=true.
- **Impact:** Effective optimization policy may say observability=true while actual emission remains disabled; cross-link [`OBSERVABILITY_EVIDENCE`](../../project/architecture/OBSERVABILITY_EVIDENCE.md).
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOKEN_OPTIMIZATION-05

- **Severity:** MEDIUM
- **Category:** RECEIPT IDENTITY / AUDIT PROVENANCE
- **Status at publication:** ACCEPTED
- **Remediation block:** TOKEN-MEASUREMENT-EVIDENCE-INTEGRITY
- **Claim falsified:** Transformation/content fingerprint is distinct from execution receipt/evidence identity.
- **Observation:** `CompressionReceipt` can contain attribution, `run_id`, `step_id`, and tenant attribution. Default `_derive_receipt_id()` hashes only `source_type`, `original_hash`, `optimized_hash`, and `strategy_id` / decision. It does not include run, step, tenant, or another execution evidence identity. The same transformation performed independently in multiple tenants/runs therefore produces the same `receipt_id`.
- **Location:**
  - `intergrax/runtime/token_optimization/receipts.py` — `CompressionReceipt`, `_derive_receipt_id()`
- **Reproduction:** Perform identical transformation in two runs/tenants; compare `receipt_id` — identical despite distinct execution scope.
- **Impact:** Audit provenance and evidence correlation collapse across executions.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOKEN_OPTIMIZATION-06

- **Severity:** MEDIUM
- **Category:** DOCUMENTATION / LIFECYCLE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** TOKEN-DOCUMENTATION-LIFECYCLE-INTEGRITY
- **Claim falsified:** Architecture and plan expose one current lifecycle truth for TOKEN-10G/H/I.
- **Observation:** At audited SHA the Token Optimization architecture header says TOKEN-10G READY_FOR_REVIEW and TOKEN-10H PLANNED / NOT STARTED. The owning feature plan at the same SHA says TOKEN-10G CLOSED; TOKEN-10H CLOSED NOT QUALIFIED (`MODEL_BEHAVIOR_MISMATCH`, 14/16, STABLE); TOKEN-10I BLOCKED_HARDWARE_CAPACITY_FINAL. The plan correctly preserves the failed qualification rather than claiming success, but architecture current-state is stale.
- **Location:**
  - `docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md` — status header (pre-sync)
  - `docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md` — authoritative TOKEN-10G/H/I rows
- **Reproduction:** Compare architecture status header with plan closeout rows at `audited_sha`.
- **Impact:** Operators and agents misread TOKEN-10H qualification and TOKEN-10I block posture.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Token Optimization is cross-layer platform capability, not LKW-owned logic | NOT falsified |
| LKW remains later product client/proof | NOT falsified |
| No second tokenizer architecturally intended | NOT falsified |
| Required pipeline-layer failures rollback net content to original and mark pipeline incomplete | NOT falsified |
| Malformed required layer results fail safe | NOT falsified |
| LLM router selects only catalog-backed pre-approved configuration IDs | NOT falsified |
| Deterministic compilation re-checks policy after LLM selection | NOT falsified |
| Policy/profile OFF block optimization | NOT falsified |
| Low-confidence router decisions block | NOT falsified |
| Unsupported source/config combinations block | NOT falsified |
| Lossy configurations blocked when `allow_lossy=false` | NOT falsified |
| Protected lossy content can require review | NOT falsified |
| Explicit SECURITY_WARNING path forces REVIEW_REQUIRED | NOT falsified |
| Cache-aware runtime reconciles cache evidence before orchestration | NOT falsified |
| Safe runtime/orchestration reports exclude raw content | NOT falsified |
| Token Optimization metadata sanitization removes raw prompts/documents/evidence/tool args | NOT falsified |
| TOKEN-10H remains honestly NOT QUALIFIED rather than promoted | NOT falsified |
| TOKEN-10I remains hardware-capacity blocked | NOT falsified |
| Findings harden existing capability; no second engine required | NOT falsified |

## Historical TOKEN delivery vs residual Protocol-v2 findings

Historical **Done/Closed** TOKEN rows (TOKEN-1..9 foundation, TOKEN-10E/F/F-EVIDENCE-EXTENSION closure, TOKEN-10G evaluation boundary, TOKEN-10H qualification evidence with honest NOT QUALIFIED outcome, TOKEN-10I hardware block) remain valid delivery facts — real contracts, pipeline, router, cache-aware runtime, proof harness, and gate evaluation were delivered as claimed. The six accepted Protocol-v2 findings document **residual measurement integrity, protected-region validation, lossy safety bypass, observability disconnect, receipt identity, and architecture lifecycle drift** at `audited_sha`. Remediation hardens the existing Token Optimization stack; it does **not** reopen closed historical rows, convert TOKEN-10H NOT QUALIFIED into success, or require a second optimization engine.

## Root-cause remediation grouping

### TOKEN-MEASUREMENT-EVIDENCE-INTEGRITY — canonical measurement, receipt identity, observability authority

**Findings:** `AUDIT-20260818-TOKEN_OPTIMIZATION-01`, `04`, `05`

Token savings, receipts, and emitted evidence use truthful canonical measurement units, execution-scoped identity, and the canonical Observability path. Cross-link [`LLM_ADAPTERS`](../../project/architecture/LLM_ADAPTERS.md) and [`OBSERVABILITY_EVIDENCE`](../../project/architecture/OBSERVABILITY_EVIDENCE.md). Do not create a private Token Optimization telemetry bus.

### TOKEN-PROTECTED-TRANSFORMATION-INTEGRITY — occurrence-aware preservation and mandatory validation

**Findings:** `AUDIT-20260818-TOKEN_OPTIMIZATION-02`, `03`

Protected-content preservation is occurrence/structure aware and the platform minimum validation cannot be disabled by lossy plugin/policy configuration. Cross-link [`CONTEXT_ENGINEERING`](../../project/architecture/CONTEXT_ENGINEERING.md), [`TOOLS`](../../project/architecture/TOOLS.md), [`RAG`](../../project/architecture/RAG.md), [`MEMORY`](../../project/architecture/MEMORY.md) where source-specific validators are owned.

### TOKEN-DOCUMENTATION-LIFECYCLE-INTEGRITY — architecture/plan current-state parity

**Findings:** `AUDIT-20260818-TOKEN_OPTIMIZATION-06`

Feature architecture and plan carry the same TOKEN-10G/H/I current state without rewriting historical qualification evidence.

## Cross-links to existing remediation

| Existing block | Relationship |
|----------------|--------------|
| **LLM-FIX-*** / **LLM_ADAPTERS** | Canonical tokenizer/provider measurement authority for TOK-01 |
| **OBS-EVIDENCE-DURABILITY-INTEGRITY** / **OBSERVABILITY_EVIDENCE** | Canonical HOS emission boundary for TOK-04 — coordinate rather than duplicate |
| **CE-CONTRACT-ACCOUNTING-INTEGRITY** / **CONTEXT_ENGINEERING** | Source-specific structural validators for protected content |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `061e03f6dc6160d8f857fbda29d1d6848d040a8d`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical TOKEN closeout rows remain valid delivery facts — not rewritten.
- Feature ownership preserved under `docs/project/capabilities/` — no domain-layer TOKEN_OPTIMIZATION plan created.

## Open questions / blocked items

- Finding 01: MEASURE_ONLY path adapter availability when no LLM adapter in scope — deferred to remediation design.
- Finding 02: per-source structural validator ownership matrix — deferred to cross-domain remediation.
- TOKEN-10I hardware capacity block remains external to Protocol-v2 findings.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-TOKEN_OPTIMIZATION-01` … `AUDIT-20260818-TOKEN_OPTIMIZATION-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
