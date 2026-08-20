# CONTEXT_ENGINEERING — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** CONTEXT_ENGINEERING
- **Constituent domains:** CONTEXT_ENGINEERING (ContextEngine · ContextCompiler · ContextPlanner · plugin registry · assembly policy)
- **Tier(s):** Tier-0 `intergrax/context/` · Tier-1 `intergrax/runtime/nexus/context/` · Tier-1 `intergrax/runtime/policy/context_assembly_policy.py`
- **audited_sha:** `86a153dac51529d4dfbf4edd0f684dacb689ae8a`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/CONTEXT_ENGINEERING.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/CONTEXT_ENGINEERING.md`
- **Scope in:**
  - `ContextAssemblyRequest` / `ContextFragment` / `ContextDecisionSnapshot` contracts
  - `run_pre_context_policy_gate()` PRE-COLLECT vs POST-COLLECT semantics
  - `DefaultContextRanker` quality gate vs mandatory/required context
  - `ContextPluginRegistry` extension surfaces vs `DefaultNexusContextEngine` execution
  - provider `supported_sources` vs emitted fragment `source` provenance
  - `ContextCompiler._enforce_hard_budget()` / `ContextCompileResult.total_tokens` accounting
  - `ContextPlanner` / UCL `resolve_ucl_context_plan()` spine (positive control)
  - Memory / RAG / UCL ownership boundaries (positive control)
  - historical CE-EXT / CE-ALIGN / CE-PROV-WIRE **Done** delivery states (positive control — not re-audited as failures)
- **Scope out:**
  - remediation implementation
  - second Context Engineering subsystem
  - universal re-qualification of every execution surface beyond documented hot paths
  - Memory or RAG domain re-audit beyond CE touchpoints
  - silent runtime fixes in production source
- **Prior audit reference(s):** Protocol v2 [`MEMORY`](MEMORY.md) (Memory/RAG/CE separation — positive control); historical CE-EXT / CE-ALIGN / CE-PROV-WIRE **Done** rows remain valid delivery facts
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `8ec504fd79223d178688fd2dd99627ba26ab9a67`

## Executive summary

**Verdict: FAIL.** Six accepted findings (4 HIGH, 2 MEDIUM) show Context Engineering contract defects on pre-collect required-source policy (unusable `required_sources`), ranker silent omission of mandatory/required context, unbound provider-to-source provenance, registry extension surfaces silently ignored by the shipped engine, false `ContextCompileResult.total_tokens` accounting, and fail-late assembly request/decision snapshot validation. Positive controls: CE / Memory / RAG / UCL responsibility split is sound; `ContextPlanner` has deterministic group identity and protected/required semantics; UCL context-plan resolution is fail-closed; adapter-aware final preflight is present; the shipped engine performs real collect → dedup → rank → plan → UCL → compile → validate → preflight → emit flow; architecture honestly states I3/P2 and uneven hot-path coverage; TOKEN-CE-1B / TOKEN-CE-2 remain planned; `DURABLE_COMPACTION` runtime execution remains not implemented; no second CE subsystem is required. Residual defects are pre-planner/plugin contract gaps distinct from the strong ContextPlanner/UCL spine — remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-CONTEXT_ENGINEERING-01

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT / POLICY CONTRACT
- **Status at publication:** ACCEPTED
- **Remediation block:** CE-POLICY-SOURCE-INTEGRITY
- **Claim falsified:** Pre-collect policy validation does not require collected evidence that cannot exist yet; post-collect gate enforces required source presence separately.
- **Substance:** `ContextAssemblyRequest` exposes `required_sources`. `DefaultNexusContextEngine` invokes `run_pre_context_policy_gate(request)` before provider collection. `run_pre_context_policy_gate()` defaults `collected=()` and always validates every required source against `present_sources`. Therefore any request with non-empty `required_sources` fails before providers have an opportunity to collect the required source. The engine later correctly invokes the same gate after collect, but the initial call makes the contract unusable.
- **Evidence:**
  - `intergrax/context/contracts.py` — `ContextAssemblyRequest.required_sources`
  - `intergrax/runtime/nexus/context/context_engine.py` — pre-collect `run_pre_context_policy_gate(request)` before collect
  - `intergrax/runtime/policy/context_assembly_policy.py` — `run_pre_context_policy_gate()` with `collected=()` default; required-source check against `present_sources`
- **Confidence:** HIGH — direct ordering and default empty collected set.
- **Target invariant:** Separate structural PRE-COLLECT validation from POST-COLLECT source-presence enforcement. PRE-COLLECT: validate request policy shape / contradictions; do not require collected evidence that cannot exist yet. POST-COLLECT: enforce required source presence and exclusions. Preserve one policy module; do not create parallel gates.

### AUDIT-20260818-CONTEXT_ENGINEERING-02

- **Severity:** HIGH
- **Category:** POLICY / MANDATORY-CONTEXT DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** CE-POLICY-SOURCE-INTEGRITY
- **Claim falsified:** No stage may silently delete mandatory/required context; quality evaluation may reject quality but must fail explicitly, not omit.
- **Substance:** `ContextFragment` exposes `mandatory`. `ContextPlanner` correctly translates mandatory fragments, plus canonical required sources, into required/protected source groups and fails when mandatory context alone exceeds the global model budget. But `DefaultContextRanker` applies its quality gate before `ContextPlanner` and does not special-case mandatory fragments. A mandatory fragment can therefore be excluded for `quality_threshold` and never reach the planner. Additionally `required_sources` are checked after collect but before ranking, so after CONTEXT-01 is fixed a required source can still pass the post-collect gate and then disappear entirely during ranking.
- **Evidence:**
  - `intergrax/context/contracts.py` — `ContextFragment.mandatory`
  - `intergrax/context/ranker.py` — quality gate without mandatory special-case
  - `intergrax/context/planner.py` — required/protected group semantics (positive contrast)
  - `intergrax/runtime/nexus/context/context_engine.py` — rank before plan ordering
  - `intergrax/runtime/policy/context_assembly_policy.py` — post-collect required-source check before ranking
- **Confidence:** HIGH — explicit pipeline ordering and ranker omission path.
- **Target invariant:** No stage may silently delete mandatory/required context. Quality evaluation may reject the quality of mandatory context, but the result must be an explicit governed assembly failure, not omission. After every stage capable of dropping fragments, final assembly must prove all required sources remain represented and all mandatory fragments required by policy remain preserved, or fail explicitly. Reuse `ContextPlanner` required/protected semantics.

### AUDIT-20260818-CONTEXT_ENGINEERING-03

- **Severity:** HIGH
- **Category:** SOURCE AUTHORITY / PROVENANCE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** CE-POLICY-SOURCE-INTEGRITY
- **Claim falsified:** Provider identity authorizes emitted `ContextFragmentSource`; provenance retains provider attestation.
- **Substance:** `ContextSourceProvider` exposes `provider_id` and `supported_sources`. `ContextPluginRegistry.add_provider()` validates only `provider_id`. `DefaultNexusContextEngine` accepts every `ContextFragment` returned by a provider without checking `fragment.source in provider.supported_sources`. Therefore a custom provider can declare `CUSTOM` but return fragments claiming `POLICY_OVERLAY`, `SYSTEM_INSTRUCTIONS`, `TASK_MESSAGE`, or another source. This is authority-relevant because `ContextPlanner` automatically treats `POLICY_OVERLAY` / `SYSTEM_INSTRUCTIONS` / `TASK_MESSAGE` as required/protected. Canonical provenance is later derived from fragment-controlled `source`/`source_id`, not a provider-attested producer identity.
- **Evidence:**
  - `intergrax/context/registry.py` — `add_provider()` validates `provider_id` only
  - `intergrax/context/providers/builtin.py` — `supported_sources` declaration pattern
  - `intergrax/runtime/nexus/context/context_engine.py` — accepts provider fragments without source authorization check
  - `intergrax/context/planner.py` — automatic required/protected treatment of policy/system/task sources
- **Confidence:** HIGH — no binding between provider authority and emitted source.
- **Target invariant:** Provider identity and emitted source authority are bound. Every collected fragment must be proven to originate from a provider authorized for that source type. Provider ID must remain available in provenance/audit evidence. Consider stronger qualification for policy/system source providers if existing plugin governance supports it; do not invent duplicate plugin trust machinery.

### AUDIT-20260818-CONTEXT_ENGINEERING-04

- **Severity:** HIGH
- **Category:** ARCHITECTURE / EXTENSIBILITY / POLICY ENFORCEMENT DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** CE-EXTENSION-RUNTIME-INTEGRITY
- **Claim falsified:** Canonical extension surface and runtime behavior match; configured validator/ranker/allocator is never silently ignored.
- **Substance:** `ContextPluginRegistry` exposes `set_ranker` / `ranker`, `set_allocator` / `allocator`, `set_formatter` / `formatter`, `set_validator` / `validator`. Public CE architecture also describes ContextPlugin bundles as providers plus optional ranker/allocator/formatter/validator. `DefaultNexusContextEngine` consumes only `registry.formatter`. It always executes constructor-owned `self._ranker`, `ContextPlanner`, and `self._validator`, and does not consume `registry.ranker`, `registry.allocator`, or `registry.validator`. Thus a host/plugin can successfully register a custom validator/ranker/allocator that is silently never executed.
- **Evidence:**
  - `intergrax/context/registry.py` — ranker/allocator/formatter/validator registration surfaces
  - `intergrax/runtime/nexus/context/context_engine.py` — uses `registry.formatter` only; constructor-owned ranker/validator
  - `intergrax/runtime/nexus/context/context_validator.py` — default validator path (positive contrast for shipped default)
- **Confidence:** HIGH — registry API vs engine consumption mismatch.
- **Target invariant:** Canonical extension surface and runtime behavior must match. Either these extension points are supported → shipped engine executes them with explicit ordering/contracts, or they are not supported → remove them from canonical registry/public claims. A configured policy/safety validator must never be silently ignored. Do not add a second CE engine to solve this.

### AUDIT-20260818-CONTEXT_ENGINEERING-05

- **Severity:** MEDIUM
- **Category:** ACCOUNTING / CONTRACT INTEGRITY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** CE-CONTRACT-ACCOUNTING-INTEGRITY
- **Claim falsified:** `ContextCompileResult.total_tokens` always reports actual computed tokens.
- **Substance:** `ContextCompiler._enforce_hard_budget()` intentionally preserves the final user message. If that message alone exceeds budget, the compiler can return messages whose actual token estimate remains above budget. `compile()` then reports `total_tokens=min(final_tokens, budget_tokens)`. Thus `ContextCompileResult` can claim `total_tokens == budget` while the returned message set actually exceeds it. Canonical hot paths run final adapter-aware preflight, so this is not proven as a model-window bypass. The defect is false accounting / diagnostics and unsafe for any consumer relying on `ContextCompileResult` before preflight.
- **Evidence:**
  - `intergrax/runtime/nexus/context/context_compiler.py` — `_enforce_hard_budget()`; `compile()` `total_tokens=min(final_tokens, budget_tokens)`
  - `intergrax/runtime/nexus/context/context_preflight.py` — adapter-aware preflight (positive control — ultimate window boundary)
- **Confidence:** HIGH — explicit min cap on reported total.
- **Target invariant:** `ContextCompileResult.total_tokens` always reports actual computed tokens. If required/current-turn content alone cannot fit: fail explicitly, or expose an explicit overflow result/state, but never cap the reported number to budget. Preserve final adapter-aware `verify_context_preflight` as the ultimate window boundary.

### AUDIT-20260818-CONTEXT_ENGINEERING-06

- **Severity:** MEDIUM
- **Category:** CONTRACT VALIDATION GAP
- **Status at publication:** ACCEPTED
- **Remediation block:** CE-CONTRACT-ACCOUNTING-INTEGRITY
- **Claim falsified:** Context request and decision snapshots are fail-fast canonical contracts.
- **Substance:** `ContextAssemblyRequest.__post_init__` currently validates `execution_scope` type but does not validate non-empty `trace_id`, `run_id`, `task_id`, `tenant_id`, `objective`, or `required_sources` vs `excluded_sources` contradiction. `ContextDecisionSnapshot` does not validate `max_memory_entries_in_context`. Builtin LTM provider consumes `max_memory_entries_in_context` directly as its `max_entries` limit.
- **Evidence:**
  - `intergrax/context/contracts.py` — `ContextAssemblyRequest.__post_init__`; `ContextDecisionSnapshot`
  - `intergrax/context/providers/builtin.py` — LTM provider uses `max_memory_entries_in_context` as limit
- **Confidence:** HIGH — missing validators on identity/source-policy/resource fields.
- **Target invariant:** Context request and decision snapshots are fail-fast canonical contracts. Use existing `TaskId`/`RunId` identity validators where semantically compatible. At minimum enforce non-empty canonical identity/scope values, disjoint required/excluded source sets, and bounded non-negative memory-entry limits. Do not introduce independently writable duplicate identity fields beyond the existing contract.

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| CE / Memory / RAG / UCL responsibility split | NOT falsified — sound |
| `ContextPlanner` deterministic group identity and source allocation | NOT falsified |
| Tool-call groups receive explicit integrity protection | NOT falsified |
| Required/protected groups participate in mandatory-context budget checks | NOT falsified |
| UCL context-plan resolution is fail-closed | NOT falsified |
| Final compiler mutation after UCL plan detected via message hash / budget / degradation checks | NOT falsified |
| Adapter-aware final context preflight is present | NOT falsified |
| `ContextBudgetSnapshot` validates positive budget values | NOT falsified |
| Shipped engine performs collect → dedup → rank → plan → UCL → compile → validate → preflight → emit | NOT falsified |
| Architecture honestly states I3/P2 and uneven hot-path coverage | NOT falsified |
| TOKEN-CE-1B / TOKEN-CE-2 remain planned — not falsely claimed shipped | NOT falsified |
| `DURABLE_COMPACTION` runtime execution remains explicitly not implemented | NOT falsified |
| No second Context Engineering subsystem required | NOT falsified |

## Strong spine vs residual contract defects

The accepted findings do **not** falsify the ContextPlanner/UCL assembly spine: deterministic planning, protected/required group semantics, UCL plan resolution, and adapter-aware preflight remain sound positive controls. Residual defects cluster **before planning** (pre-collect policy gate, ranker omission) and on **plugin/registry contract surfaces** (ignored extensions, unbound provider source authority) plus **compile-result accounting** and **request/snapshot validation**. Remediation groups preserve the single engine and single global input budget authority.

## Root-cause remediation grouping

### CE-POLICY-SOURCE-INTEGRITY — required/mandatory source policy and trusted provenance

**Findings:** `AUDIT-20260818-CONTEXT_ENGINEERING-01`, `AUDIT-20260818-CONTEXT_ENGINEERING-02`, `AUDIT-20260818-CONTEXT_ENGINEERING-03`

Required/mandatory source policy and trusted provider-to-source provenance form one fail-closed assembly authority.

### CE-EXTENSION-RUNTIME-INTEGRITY — registry extension contracts match execution

**Findings:** `AUDIT-20260818-CONTEXT_ENGINEERING-04`

Registry extension contracts and actual `ContextEngine` execution semantics are identical; no accepted-but-ignored validator/ranker/allocator configuration.

### CE-CONTRACT-ACCOUNTING-INTEGRITY — truthful accounting and fail-fast contracts

**Findings:** `AUDIT-20260818-CONTEXT_ENGINEERING-05`, `AUDIT-20260818-CONTEXT_ENGINEERING-06`

Truthful token accounting plus fail-fast typed assembly identity/configuration.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `86a153dac51529d4dfbf4edd0f684dacb689ae8a`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical CE-EXT / CE-ALIGN / CE-PROV-WIRE plan **Done** rows remain valid delivery facts — not rewritten.

## Open questions / blocked items

- 04: choose supported extension execution vs registry surface reduction — operator decision deferred to remediation.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 6 (`AUDIT-20260818-CONTEXT_ENGINEERING-01` … `AUDIT-20260818-CONTEXT_ENGINEERING-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
