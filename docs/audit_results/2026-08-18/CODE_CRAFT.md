# CODE_CRAFT — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** CODE_CRAFT
- **Constituent domains:** CODE_CRAFT (Ephemeral Code Craft — orchestrator · profile · static gate · sandbox · promotion · ephemeral registry)
- **Tier(s):** Tier-0 `intergrax/codecraft/` · Tier-1 `intergrax/runtime/codecraft/` · Tier-0 `intergrax/tools/providers/codecraft/` · Tier-3 `CodeCraftProfile` wiring
- **audited_sha:** `f985ad342d0d6db38c9998df67f9cd7bc10bfa46`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 2 CRITICAL / 5 HIGH / 0 MEDIUM / 0 LOW
- **Operator decision:** all 7 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/CODE_CRAFT.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/CODE_CRAFT.md`
- **Scope in:**
  - `CodeCraftProfile` typed contract and `resolve_codecraft_profile()` task override behavior
  - `CodeCraftOrchestrator` lifecycle: start / iterate / run / get_state / promote / dispose
  - `CodeCraftSessionManager` session storage and identity binding
  - `EphemeralToolRegistryStore` session-scoped registry
  - `resolve_craft_sandbox_session()` isolation-tier resolution and fallback
  - `StaticCodeGate` pre-execution scanning
  - `CraftTestRunner` verification substrate binding
  - `CraftResultPromoter` promotion eligibility and evidence
  - `codecraft.*` tool provider inputs (`hitl_approved`, tenant/task/run metadata, caller-supplied `craft_id`)
  - CodeCraft / Sandbox / Tools / Governance / CVL ownership separation
  - ECC-0…ECC-6 and S7–S11 historical closeout (positive control — not re-audited as failures)
- **Scope out:**
  - remediation implementation
  - second code-generation runtime design
  - full Governed Execution / UER re-audit beyond CodeCraft HITL touchpoints
  - universal hostile-code / sandbox-escape production proof
  - CVL semantic judge depth beyond craft-loop integration touchpoints
- **Prior audit reference(s):** ECC-0…ECC-6 **Done**; S7–S11 post-closeout **Done**; Full Harness LC internal evidence; Protocol v2 [`TOOLS`](TOOLS.md) (governed tool boundary); [`POLICY_GOVERNANCE`](POLICY_GOVERNANCE.md) (PG-FIX scoped approval); [`IDENTITY_TRUST`](IDENTITY_TRUST.md) (execution identity closure)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** —

## Executive summary

**Verdict: FAIL.** Seven accepted findings (2 CRITICAL, 5 HIGH) show CodeCraft session authority keyed only by `craft_id` without runtime-trusted tenant/task/run ownership validation; HITL approval accepted from caller-controlled tool input and divergent between iterative and `codecraft.run` paths; task metadata may escalate host `disabled` posture to executable modes; promotion fabricates success evidence without eligibility checks; verification re-resolves a different sandbox than execution; required cloud/container isolation silently downgrades to local; and `network_egress` is profile contract without runtime enforcement. Positive controls: conceptual CodeCraft / Sandbox / Tools / Governance / CVL split remains sound; `CodeCraftOrchestrator` is the intended canonical lifecycle owner; static gate runs before execution on audited paths; profile is typed extra-forbid; iteration/time budgets exist; ephemeral tools are separated from global `ToolRegistry`; architecture already documents cloud/local fallback and partial egress honestly — findings 06/07 are architecture defects, not documentation concealment.

## Verdict

**FAIL** — 2 CRITICAL / 5 HIGH / 0 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-CODE_CRAFT-01

- **Severity:** CRITICAL
- **Category:** SECURITY / TENANT ISOLATION / IDENTITY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY
- **Claim falsified:** CodeCraft session authority is bound to canonical execution identity; every stateful operation validates ownership using runtime-trusted tenant/task/run identity; `craft_id` alone is not an authorization capability.
- **Substance:** `CodeCraftSession` carries `tenant_id`/`task_id`, but `CodeCraftSessionManager` stores and retrieves sessions only by `craft_id`. Default manager is a process-global `_default_manager`. `get_state`, `iterate`, `promote`, and `dispose` resolve by `craft_id` without verifying that the caller's canonical tenant/task/run identity owns the session. Tool inputs expose `tenant_id`/`task_id`/`agent_id`/`run_id` as caller-controlled fields. `craft_id` may be caller-supplied on start, and `open()` silently assigns `self._sessions[sid] = session`, allowing an existing id to be overwritten. `EphemeralToolRegistryStore` is likewise keyed only by `craft_id`.
- **Evidence:**
  - `intergrax/runtime/codecraft/session_manager.py` — `_sessions` keyed by `craft_id` only; `open()` overwrite semantics
  - `intergrax/tools/providers/codecraft/contracts.py` — caller-controlled identity fields on tool inputs
  - `intergrax/runtime/codecraft/ephemeral_registry.py` — `craft_id`-only registry keying
- **Confidence:** HIGH — direct code path; no ownership validation on stateful operations.
- **Target invariant:** CodeCraft session authority must be bound to canonical execution identity, not caller-asserted metadata. Opening an already-existing identity in a conflicting scope must fail closed. Reuse existing Intergrax execution identity contracts.

### AUDIT-20260818-CODE_CRAFT-02

- **Severity:** CRITICAL
- **Category:** GOVERNANCE / HITL AUTHORIZATION BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY
- **Claim falsified:** HITL approval is authoritative execution state, never an LLM/tool argument; `codecraft.run` and start/iterate converge on the same approval boundary.
- **Substance:** `CodeCraftIterateToolInput` contains caller-controlled `hitl_approved: bool`. `codecraft_iterate` passes it directly to `CodeCraftOrchestrator.iterate()`. The orchestrator treats this boolean as sufficient approval and persists `session.hitl_approved`. Unit tests explicitly "approve" supervised execution by sending `hitl_approved=True`. Separately, `codecraft.run` executes supervised / `require_hitl_before_exec` profiles without performing the CodeCraft HITL check used by the iterative path.
- **Evidence:**
  - `intergrax/tools/providers/codecraft/contracts.py` — `hitl_approved` on iterate input
  - `intergrax/tools/providers/codecraft/service.py` — passes `hitl_approved` to orchestrator
  - `intergrax/runtime/codecraft/orchestrator.py` — boolean approval persistence
  - `tests/unit/runtime/codecraft/test_orchestrator.py` — test approves via `hitl_approved=True`
  - `tests/unit/tools/providers/codecraft/test_codecraft_run.py` — run path without HITL check parity
- **Confidence:** HIGH — caller self-assertion is the approval mechanism on iterative path; run path diverges.
- **Target invariant:** Only canonical Governed Execution / UER approval evidence scoped to the exact task/run/craft/action may authorize generated-code execution. Remove caller self-assertion.

### AUDIT-20260818-CODE_CRAFT-03

- **Severity:** HIGH
- **Category:** AUTHORIZATION / CONFIGURATION ESCALATION
- **Status at publication:** ACCEPTED
- **Remediation block:** CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY
- **Claim falsified:** Task/run overrides may only narrow host-authorized CodeCraft posture unless a separate trusted policy authority explicitly approves expansion.
- **Substance:** `resolve_codecraft_profile()` reads `task_metadata.codecraft_mode` and applies it with `profile.model_copy(update={"mode": mode_override})`. There is no check that the override is permitted by the host profile or only narrows capability. Existing test demonstrates a base mode `disabled` becoming `supervised`. The same mechanism permits `disabled` → `autonomous`.
- **Evidence:**
  - `intergrax/applications/_shared/codecraft_wiring.py` — `resolve_codecraft_profile()` override application
  - `intergrax/codecraft/profile.py` — mode lattice without escalation guard
- **Confidence:** HIGH — unconditional metadata override without narrowing lattice.
- **Target invariant:** Host `disabled` cannot become executable CodeCraft from task metadata. Model the allowed override lattice explicitly and fail fast on escalation.

### AUDIT-20260818-CODE_CRAFT-04

- **Severity:** HIGH
- **Category:** VERIFICATION / PROMOTION INTEGRITY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** CODECRAFT-VERIFICATION-INTEGRITY
- **Claim falsified:** Promotion is an evidence-consuming state transition; only promotion-eligible verified sessions may be promoted; promotion must never fabricate passed gate/verdict.
- **Substance:** `CodeCraftOrchestrator.promote(craft_id)` retrieves a session and calls `CraftResultPromoter` without checking promotion eligibility: last iteration verdict, static gate, execution success, required test result, required HITL, session lifecycle/status. `CraftResultPromoter` constructs `success=True`, `StaticGateResult(passed=True)`, `verdict="promote"` independently of actual session evidence. `promotion_schema_ref` does not resolve/validate a custom schema; both schema-ref and no-schema paths return `payload.model_dump()`.
- **Evidence:**
  - `intergrax/runtime/codecraft/orchestrator.py` — `promote()` without eligibility gate
  - `intergrax/codecraft/promoter.py` — fabricated success/gate/verdict
- **Confidence:** HIGH — promotion path does not consume session verification evidence.
- **Target invariant:** If `promotion_schema_ref` is configured, failure to resolve or validate it must fail closed. Reuse canonical CVL/verdict and schema registry contracts where available.

### AUDIT-20260818-CODE_CRAFT-05

- **Severity:** HIGH
- **Category:** VERIFICATION / ISOLATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** CODECRAFT-VERIFICATION-INTEGRITY
- **Claim falsified:** Verification/tests for a craft iteration execute against the same immutable artifact and the same sandbox/revision identity as the execution being judged.
- **Substance:** `CodeCraftOrchestrator` resolves craft execution sandbox with `resolve_craft_sandbox_session()` and executes code through an `exec_ctx` containing that resolved sandbox. It then runs `CraftTestRunner(profile).run(self._ctx, ...)` using the original context. `CraftTestRunner` independently calls `resolve_sandbox_session(ctx)` rather than using the exact sandbox session used for craft execution. For cloud/container paths this can make execution and tests occur in different substrates/environments.
- **Evidence:**
  - `intergrax/runtime/codecraft/orchestrator.py` — `exec_ctx` vs `_ctx` for test runner
  - `intergrax/codecraft/test_runner.py` — independent `resolve_sandbox_session(ctx)`
- **Confidence:** HIGH — test substrate re-resolution diverges from execution substrate.
- **Target invariant:** Do not silently re-resolve a different local sandbox during verification.

### AUDIT-20260818-CODE_CRAFT-06

- **Severity:** HIGH
- **Category:** ARCHITECTURE / SECURITY ISOLATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** CODECRAFT-ISOLATION-INTEGRITY
- **Claim falsified:** Isolation tier is a minimum security requirement; a required cloud/container tier must fail closed when an eligible substrate cannot be resolved.
- **Substance:** `resolve_craft_sandbox_session()` attempts hosted resolution for cloud/container but falls through to `resolve_sandbox_session(ctx)` when none resolves. Thus a profile explicitly requiring cloud/container may execute locally. Unit test explicitly validates cloud-tier local fallback. Architecture documents this limitation honestly; Protocol v2 records it as an accepted architecture defect.
- **Evidence:**
  - `intergrax/runtime/codecraft/sandbox_resolver.py` — local fallback on unresolved hosted substrate
  - `tests/unit/runtime/codecraft/test_orchestrator.py` — cloud-tier local fallback test
- **Confidence:** HIGH — silent downgrade is intentional as-built behavior.
- **Target invariant:** The regulated CodeCraft preset must not silently downgrade to local unless an explicit trusted policy defines an allowed downgrade.

### AUDIT-20260818-CODE_CRAFT-07

- **Severity:** HIGH
- **Category:** SECURITY / PAPER CONTROL
- **Status at publication:** ACCEPTED
- **Remediation block:** CODECRAFT-ISOLATION-INTEGRITY
- **Claim falsified:** `network_egress` is enforceable runtime policy; `deny` must result in a sandbox substrate/network policy that can prove outbound network denial before generated code is executed.
- **Substance:** `CodeCraftProfile` exposes `network_egress = deny | allowlist`; regulated preset uses `deny`. Current implementation references this field in profile/wiring/documentation but does not enforce it in sandbox resolution/execution. Architecture states runtime egress enforcement is partial.
- **Evidence:**
  - `intergrax/codecraft/profile.py` — `network_egress` field
  - `intergrax/applications/_shared/codecraft_wiring.py` — wiring references without enforcement
  - `intergrax/runtime/codecraft/sandbox_resolver.py` — no egress posture binding
- **Confidence:** HIGH — field is contract-only on audited paths.
- **Target invariant:** If selected substrate cannot satisfy requested egress posture, fail closed. Bind enforcement evidence to substrate capability.

## Why findings 01 and 02 are CRITICAL

**01 — tenant/session authority:** Stateful craft operations (`get_state`, `iterate`, `promote`, `dispose`) are reachable by `craft_id` alone while identity fields on tool inputs are caller-asserted. In a multi-tenant or multi-task runtime this is a direct cross-tenant/cross-task session access primitive: any caller who learns or guesses a `craft_id` can read, mutate, promote, or dispose another tenant's generated-code session. `craft_id` is treated as a capability token but is not bound to runtime-trusted execution identity. Session overwrite on `open()` compounds the defect. This is a security isolation failure on the generated-code authority plane — not a quality or ergonomics gap.

**02 — HITL authorization bypass:** Supervised and HITL-gated execution is a primary safety control for generated code. Accepting `hitl_approved` from tool input lets the LLM (or any tool caller) self-authorize execution without canonical Governed Execution / UER approval evidence. Divergence between iterative and `codecraft.run` paths means the governance boundary is not uniform. This defeats the human gate for a subsystem whose entire purpose is executing untrusted synthesized code — equivalent to a governance bypass on a safety-critical boundary.

## Positive controls / falsification log

1. **CodeCraft / Sandbox / Tools / Governance / CVL ownership split conceptually sound** — not falsified; boundaries documented and observed on primary paths.
2. **CodeCraftOrchestrator canonical lifecycle owner** — not falsified; single orchestrator entry for start/iterate/run/promote/dispose.
3. **StaticCodeGate runs before execution** — not falsified on audited orchestrator/run paths.
4. **StaticCodeGate validates language, size, Python syntax, forbidden imports/calls** — not falsified.
5. **CodeCraftProfile typed and extra-forbid** — not falsified.
6. **`max_iterations` / `max_total_exec_time_s` controls exist** — not falsified.
7. **Ephemeral tools separated from global ToolRegistry** — not falsified.
8. **Architecture honest about cloud/local fallback and partial egress** — not falsified; findings 06/07 are defects, not concealment.
9. **No second code-generation runtime required** — not falsified; gaps are authority, governance, verification, and isolation integrity on existing spine.
10. **ECC/S7–S11 historical closeout** — not falsified as delivery facts; this audit records Protocol v2 residual gaps on top of historical **Done** rows.

## Root-cause remediation grouping

Planning only — **audit persistence does NOT implement remediation.**

### CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY — session authority, canonical HITL, override lattice

**Findings:** 01, 02, 03

**Priority:** P0

Bind craft session authority to canonical tenant/task/run execution identity. Remove caller-controlled HITL self-assertion; converge `codecraft.run` and iterative lifecycle on the same Governed Execution / UER approval boundary. Model host/task override lattice — narrow-only unless trusted policy approves expansion.

### CODECRAFT-VERIFICATION-INTEGRITY — promotion eligibility and same-sandbox verification

**Findings:** 04, 05

**Priority:** P0

Promotion must consume real verification evidence and cannot manufacture success. Tests/verdict must bind to the same sandbox/artifact identity as execution. `promotion_schema_ref` fail-closed when configured.

### CODECRAFT-ISOLATION-INTEGRITY — anti-downgrade and egress enforcement

**Findings:** 06, 07

**Priority:** P0/P1

Required isolation tiers fail closed against silent downgrade. `network_egress` becomes runtime-enforced substrate capability with provable denial when `deny`.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `f985ad342d0d6db38c9998df67f9cd7bc10bfa46`; current `development` HEAD was not re-audited.
- Tests are supporting evidence, not standalone proof of production safety.
- Remediation not performed in this task.
- Historical ECC/S7–S11 **Done** rows remain valid delivery facts — not rewritten.

## Open questions / blocked items

- 01: exact binding to existing execution identity contracts (UER `RunId` / tenant authority) — reuse canonical owners, no parallel string identity.
- 02: approval evidence shape from Governed Execution — coordinate with PG-FIX-C scoped approval consumption.
- 06: whether regulated preset retains explicit named downgrade policy vs strict fail-closed — operator decision deferred to remediation.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 7 (`AUDIT-20260818-CODE_CRAFT-01` … `AUDIT-20260818-CODE_CRAFT-07`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
