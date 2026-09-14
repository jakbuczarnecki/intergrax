# Execution Engine — Security Adversarial & Abuse Model (EE-B3-C)

**Status:** Certification baseline (deterministic abuse-case verification).  
**Parent:** `EXECUTION_ENGINE_SECURITY_BOUNDARY_AND_THREAT_MODEL.md` (EE-B3-A).  
**Invariant:** Malicious / forged / conflicting input → canonical validation → DENY / fail-closed / typed rejection → no execution, no authority expansion, no tenant switch, no side effect.

## 1. Methodology

- **Audit-first:** B3-A defines owners and boundaries; B3-C proves they hold under abuse.
- **Deterministic:** No fuzzing, sleeps, or race-dependent exploits.
- **Real boundaries:** Authorization logic is not mocked away; external providers and stores may be faked.
- **Assertions:** Denied cases require zero execution/tool/provider side effects and stable tenant/authority where applicable.

## 2. Abuse-case matrix

| ID | Attack | Target asset | Expected control | Expected outcome | Forbidden outcome |
| -- | ------ | ------------ | ---------------- | ---------------- | ----------------- |
| AC-01 | Forged execution identity | RunId / AttemptId / ExecutionId / TaskId | `validate_*`, identity authority mint allowlist | Format reject; no active bind | Caller-minted production identity |
| AC-02 | Cross-tenant resume | Checkpoint `tenant_id` | `validate_checkpoint_identity_binding` | `REJECT_TENANT` | Use requested tenant |
| AC-03 | Child authority escalation | Parent scopes | `DefaultStrictAuthorityPolicy` / `mint_effective_delegation_authority` | `DelegationAuthorityError` | Child ⊄ parent |
| AC-04 | Governance bypass | Root admission | `RuntimeExecutionPolicyAdmissionEvaluator` | DENY when unconfigured | Silent ALLOW on supported path |
| AC-05 | Tool side-effect bypass | External effects | `RuntimeToolInvoker` + side-effect authorization | DENY; executor count 0 | Mutate without authorization |
| AC-06 | Policy decision spoofing | `EvaluatedPolicyDecision` | Bundle binding + `assert_consistent_with_bundle` | Reject mismatch | Forged ALLOW sticks |
| AC-07 | Retry tenant mutation | Lifecycle `(tenant_id, run_id)` | `ExecutionAttemptRetryService` | No cross-tenant CAS | Privilege / tenant expansion |
| AC-08 | Recovery context mutation | Resume authority / tenant | Checkpoint resume validation | Typed reject | Blind checkpoint trust |
| AC-09 | Checkpoint tampering | Revision / identity fields | `validate_canonical`, CAS store | Stale / identity reject | Corrupt resume |
| AC-10 | HITL misuse | Approval grant scope | Declarative HITL bridge | HITL still required | Grant unrelated action |
| AC-11 | Confused deputy | Provider credentials | `AgentRuntimeGovernanceBoundary` before executor | `ToolGovernanceDeniedError`; provider 0 | Provider authorizes caller |
| AC-12 | Resource abuse | Fan-out bounds | NPSC-5B `validate_fan_out_request` | `InvalidFanOutError` | Unbounded fan-out |

## 3. Identity abuse

Valid canonical **format** does not imply **authority**. Mint paths remain on `identity_authority` allowlist (EE-A2 gate).

## 4. Tenant abuse

Resume and identity binding reject tenant mismatch; no `if mismatch: use requested tenant` path in supported validation.

## 5. Authority abuse

Child effective authority ⊆ parent; overreach raises `DelegationAuthorityError` (not silent narrowing when contract requires rejection).

## 6. Governance spoofing

Untrusted callers cannot supply authoritative `EvaluatedPolicyDecision`. Root admission defaults DENY when rules are missing.

## 7. Tool / side-effect abuse

Unknown tools fail without executor entry. Mutating tools require meaningful side-effect authorization and registry resolution (no arbitrary import path in invoker).

## 8. Retry / recovery abuse

Retry preserves `run_id` per tenant lifecycle; wrong-tenant retry does not advance source tenant state. Governance-denied failures do not retry.

## 9. Checkpoint tampering

Canonical runtime validation and revision CAS reject stale or inconsistent identity trees.

## 10. Confused deputy

Core governance denies before `ToolExecutor`; production mode requires configured governance boundary.

## 11. Resource abuse

Fan-out item count and concurrency are capped (`MAX_FAN_OUT_ITEMS`, `MAX_FAN_OUT_CONCURRENCY`).

## 12. Compound abuse

Documented scenarios: authority violation before permissive policy; wrong-tenant retry with valid `run_id`; governance deny with zero provider calls.

## 13. Security bypass inventory

| Candidate | Reachable in supported prod? | Security gate present? | Verdict |
| --------- | ---------------------------: | ---------------------: | ------- |
| `AllowingRuntimeExecutionPolicyAdmission` | Tests / reference adapter only | Not wired under `intergrax/` tree | Accept (test-only) |
| `AllowingPhysicalDelegationGovernance` | Application wiring with separate governance plane | Physical delegation evaluator | Accept (scoped) |
| Direct `mint_*` outside allowlist | No (EE-A2) | AST gate | Closed |
| P0 execution bypass | No | Inventory = 0 | Closed |

## 14. Cross-session exclusions (NPSC-5F)

**Do not modify** in EE-B3-C: `causal_evidence*`, `export_boundary`, `background_execution/**`, NPSC-5F fingerprints/resignoff. Findings → handoff only.
