# SFX R2 ↔ SEF-X Cross-Program Amendment Proposal (PROPOSED)

**Status:** PROPOSED — **not** incorporated into canonical `ASF-SEF-MASTER-PLAN.md`  
**R1 binding rule:** **HO-2** — only a human submits an `EngineeringIntent`; SFX never auto-submits ([`SFX-MASTER-PLAN.md`](../../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) §3.2)  
**SEF binding rule:** **CON-N-01** — `human submit only` ([`ASF-SEF-MASTER-PLAN.md`](../../../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md) §contracts)

## 1. Problem statement

R2 bounded autonomy requires multiple L2 engineering iterations (build → test → repair) **without** per-iteration trivial human clicks, while preserving attributable human approval, ticket/plan governance, and SEF engineering FSM authority.

R1 explicitly forbids SFX auto-submit. **This proposal does not reinterpret HO-2 unilaterally.**

## 2. Proposed mechanism: scoped delegation grant

### 2.1 Actors

| Actor | Role |
| --- | --- |
| Human approver | Signs `AutonomyEnvelope` + `DelegationGrant` once (IdP-attributed `operator_id`) |
| SFX Tier-2 | Emits **derived** `EngineeringIntent` drafts within envelope; never holds submit authority |
| SEF-X EEA | Validates each intent against grant digest, ticket, budget, backend policy; **records** submit with `delegated_from_grant_ref` |
| Independent reviewer | Certifies grant semantics before OD signature |

### 2.2 Grant payload (PROPOSED fields)

```text
grant_id, issuer_operator_id, tenant_id, mission_id,
goal_digest, blueprint_family_digest (or exact pin),
ticket_ref, plan_digest,
allowed_modes=[AGENTIC_ENGINEERING],
max_submissions, max_repair_rounds_per_job (SEF bounded repair),
cost_budget_remaining_ref,
allowed_workspace_roots[], deny_egress,
expires_at, revocable_at,
signature / decision_binding_ref
```

### 2.3 Per-submit validation (SEF-side — PROPOSED CON-N-01 extension)

On each derived intent:

1. Grant signature valid, not expired, not revoked.
2. Intent text + attachments digest ⊆ grant scope (no new repos, no production refs).
3. `ticket_ref` matches grant; SEF FSM state allows intake.
4. Budget counters decrement; fail-closed on zero.
5. Attribution: `submitted_by=sef-orchestrator`, `on_behalf_of=grant.issuer_operator_id`, `grant_ref=…`.

### 2.4 HO mapping (proposed)

| HO | R1 | R2 (if amendment accepted) |
| --- | --- | --- |
| HO-1 | Intent draft from SFX | unchanged |
| HO-2 | Human submit only | Human **grant** once; SEF accepts **derived** submits under grant |
| HO-3..HO-7 | Evidence, bundle, idempotency | unchanged + `grant_ref` on evidence |

## 3. Authority boundaries (unchanged)

- SEF still owns merge/PR, repair loop, workspace lease, coding backend.
- SFX still does not invoke ToolRuntime for business execution.
- Engineering merge to `development` remains operator/human governed (no auto-merge).

## 4. Conflicts register updates (PROPOSED)

| ID | Change |
| --- | --- |
| C-SEF-01 | Add SFX as named consumer **with** delegation semantics — requires **OD-SEF-07** or new OD |
| C-SEF-02 | Egress mandatory for consumer path — unchanged urgency |
| **C-SEF-R2-01** (new) | Grant revocation race vs in-flight job — SEF must define terminal handling |

## 5. Required SEF track deliverables (not SFX-owned)

| Item | Owner |
| --- | --- |
| ADR or CON-N-01 v2 schema for delegated submit | SEF architect |
| `OD-SEF-04` (proposed): delegation FSM semantics | Operator |
| Ticket/plan approval for grant-bearing missions | SEF + SFX joint |
| GATE-SEF-CONSUME evidence extension | SEF tester |

## 6. Open questions (escalate to operator)

1. Is one grant per mission sufficient, or per candidate branch?
2. Does grant cover **repairBounded** only, or net-new candidate scaffolds?
3. Cross-tenant grants: **DENY** (recommended).

## 7. Implementation sequencing

| Phase | Behavior |
| --- | --- |
| Until OD signed | R2 code must implement **ASSISTED** only (human submit per intent) — R2-W4-06 |
| After OD + SEF schema | Enable `BOUNDED_AUTONOMOUS` flag on mission — R2-W4-07 |

## 8. Delegation semantics (15 decisions — PROPOSED)

| # | Topic | SFX (PROPOSED) | SEF (PROPOSED) | Status |
| --- | --- | --- | --- | --- |
| 1 | Mission creation | Tier-2 composes MissionRecord (L1 persist) | — | SFX-owned |
| 2 | Scope approval | Human signs CON-13 + AutonomyEnvelope digest | — | Human authority |
| 3 | Approver identity | IdP principal on CON-13 | Validates issuer on grant | Existing + PROPOSED grant fields |
| 4 | Scope vs new scope | Digest pin on goal/blueprint/envelope | Reject intent if digest ∉ grant | SEF validator |
| 5 | Derived intent verify | Draft only | Schema + grant + ticket + budget | **REQUIRES SEF OWNER DECISION** |
| 6 | New ticket/plan | Scope change → new CON-13 + grant | New ticket if plan_digest changes | Joint OD |
| 7 | Multi-iteration cost | Mission driver aggregates counters | SEF authoritative job cost | Refs only in SFX |
| 8 | Grant expiry | Driver STOP; no new handoffs | Reject new submits | Contract PROPOSED |
| 9 | Revocation | Immediately block new submits | Cancel request on active job | **No claim of instant undo of in-flight work** |
| 10 | In-flight jobs | SFX marks PAUSED | SEF cancel or run-to-safe-terminal | **REQUIRES SEF OWNER DECISION** |
| 11 | Duplicate effects | Idempotent handoff digest | HO-4/5 patterns | Existing |
| 12 | Crash resume | CW version + job ref | SEF job idempotency | Existing + mission phase |
| 13 | Tech vs business fail | Build pass ≠ business PASS | Engineering success separate from SFX verdict | SFX-INV-R2-03 |
| 14 | Stagnation stop | Envelope stop_on_stagnation_rounds | — | SFX driver |
| 15 | Audited events | Mission/envelope/iteration CWs | Engineering evidence bundle | Immutable evidence |

### Revocation (corrected)

- **Immediate:** no new EngineeringIntent submit after effective revoke timestamp.
- **Active job:** SEF issues cancel; if job non-cancellable, mark UNKNOWN_OUTCOME and reconcile — PROPOSED until SEF defines terminal states.
- **Not claimed:** instantaneous reversal of all side effects already committed by SEF/backend.
