# EQO-PKG-A4-034 — IMPLEMENTATION_BLOCKED

**Finding:** EQOA-A4-034 (EDER-F-0143)  
**Root cause id:** EQOA-A4-034  
**Base pin:** `8e146e21cd8c63a47d621edc05d5f0036ba0c9ee`  
**Status:** BLOCKED_EXACT_REASON

## Decision

Product code change deferred. Architecture and FRZ-REL-02 do not state a single qualified retry-budget owner across the coexisting LLM-provider and step-layer mechanisms; owner must remove, unify, or sanction each layer before implementers edit `intergrax/`.

## Production wiring (active)

| Mechanism | Path | Production callers @ base pin |
|-----------|------|-------------------------------|
| Inline per-logical-call budget | `intergrax/llm_adapters/_shared/resilience.py` (`_InlineRetryBudgetLogicalCall` via `_open_retry_budget`) | Used on every LLM SDK retry path when `get_llm_provider_retry_budget_port()` is `None` (default). |
| `RetryBudgetPort` registry | `intergrax/llm_adapters/_shared/provider_retry_budget.py` | `set_llm_provider_retry_budget_port` has **no** production caller. |

## Dormant modules in scope (no production imports)

| Class | Module | Non-production callers |
|-------|--------|------------------------|
| `LocalProviderRetryBudget` | `intergrax/runtime/resilience/local_provider_retry_budget.py` | `tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py` (injects port); qualification inventories (`tests/qualification/state_x/_state_x_explicit_mechanism_classifications.py`, `tests/qualification/trace_x/_trace_x_p6_restart_resume_classification.py`, `tests/qualification/ebh_6/_l3_reachability_evidence.py`). |
| `StepRetryBudget` | `intergrax/runtime/reliability/step_retry_budget.py` | `tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py`, `tests/unit/runtime/architecture/test_npsc5e_r1_execution_retry_attempt_semantics.py`, `tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py`. **Not referenced under `intergrax/` outside its defining module.** |

## Missing canonical sentence (FRZ-REL-02)

Frozen NPSC-5E retry taxonomy (`docs/project/maintainers/architecture/NPSC_5E_RECOVERY_CHECKPOINT_RETRY_ARCHITECTURE.md`) names **`StepRetryBudget`** as the step/node retry owner and lists **separate budgets (not collapsed)**, while W2 qualification artefacts name **`LocalProviderRetryBudget`** as the `RetryBudgetPort` implementation, and product LLM calls use the **inline** budget by default. None of `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md`, the NPSC-5E doc, or FRZ-REL-02 matrix prose contains one sentence that (a) designates the qualified **LLM provider** retry owner (`_InlineRetryBudgetLogicalCall` default vs `LocalProviderRetryBudget` via composition), and (b) states whether unwired `StepRetryBudget` remains the sanctioned step-layer owner or must be retired. Until that owner sentence exists, implementers must not delete `LocalProviderRetryBudget`, wire a second budget, or collapse layers.

## Blocker

Owner decision required (per `eqo_closure/runs/wb07/COR_DRV.md` EQOA-A4-034 route): wire `LocalProviderRetryBudget` at the host composition root **or** retire it relative to the inline default; separately confirm `StepRetryBudget` against the NPSC step/node row or supersede it in architecture.

## Outcome

**BLOCKED** — docs-only record; no `intergrax/` edit at this base pin.