# DECISION_SYSTEM — extended architecture

**Parent hub:** [`DECISION_SYSTEM.md`](../DECISION_SYSTEM.md)

> **Canon:** frozen target. Nexus executes Decision Lifecycle — no second runtime.

## 1. Lifecycle decomposition

| Stage | Owner | Persists |
| ----- | ----- | -------- |
| Proposal | Nexus lifecycle | Candidate Decision + Decision Version |
| Deliberation (optional) | DecisionStrategy via Nexus | Candidate versions + disagreement artifact |
| Verification | Verification Pipeline | Verification Result (+ Challenge) |
| Revision | Decision Lifecycle | New immutable Decision Version |
| Adjudication (optional) | Lifecycle + HITL invocation | Adjudication record |
| Resolution | Decision Lifecycle | ACCEPTED / REJECTED / UNRESOLVED |
| Finalization | Decision Lifecycle | Authoritative Accepted Decision **or** Resolution Record |

## 2. Artifact families

| Family | Role |
| ------ | ---- |
| Decision Artifact | Typed payload bound to a Decision Version |
| Verification Result | Stage-composed correctness verdict for one version |
| Challenge | Semantic insufficiency signal → revision (not mutation) |
| Disagreement Artifact | Structured dissent preserved through synthesis |
| Authoritative Accepted Decision | Terminal ACCEPTED binding one version |
| Authoritative Resolution Record | Terminal REJECTED / UNRESOLVED without fake acceptance |

## 3. Lifecycle invariants

```text
DS-INV-001  Candidate ≠ Authoritative — append versions, never overwrite.
DS-INV-002  Decision Resolution ≠ execution termination.
DS-INV-003  At most one terminal authoritative outcome per decision scope.
DS-INV-004  Verification checks — does not finalize or authorize alone.
DS-INV-005  Approval / authorization binds Decision ID + Version + scope + tenant + execution identity.
DS-INV-006  UNRESOLVED is a first-class auditable resolution.
```

## 4. Extension surfaces

| Surface | Contract |
| ------- | -------- |
| DecisionStrategy | Registered strategies (Single, Council, Rule, Hybrid, …) |
| Verification stage plugins | Ordered compositional stages |
| Decision Artifact kinds | Typed registration — not `dict[str, Any]` payloads |
| Adjudication hooks | Optional human / policy adjudication interfaces |

## 5. Canonical ownership

| Concern | Owner |
| ------- | ----- |
| Lifecycle orchestration | Nexus |
| Strategy semantics | DecisionStrategy plugins |
| Correctness gates | Verification Pipeline |
| Execution authorization | Policy / Governed Execution |
| Human authority records | HITL (Reliability domain) |
| Audit evidence | Observability |
| Problem classification | Diagnostics (adjacent, not owner) |

## 6. Related satellites

| Topic | Route |
| ----- | ----- |
| Identity / versioning | [`DECISION_SYSTEM_identity_version_lineage.md`](DECISION_SYSTEM_identity_version_lineage.md) |
| Lifecycle / resolution | [`DECISION_SYSTEM_lifecycle_state_resolution.md`](DECISION_SYSTEM_lifecycle_state_resolution.md) |
| Authority / finalization | [`DECISION_SYSTEM_authority_finalization.md`](DECISION_SYSTEM_authority_finalization.md) |
| Concurrency / recovery | [`DECISION_SYSTEM_concurrency_recovery.md`](DECISION_SYSTEM_concurrency_recovery.md) |
| Platform boundaries | [`DECISION_SYSTEM_platform_boundaries.md`](DECISION_SYSTEM_platform_boundaries.md) |
| Verification depth | [`DECISION_VERIFICATION.md`](../DECISION_VERIFICATION.md) · [`DECISION_VERIFICATION_extended_depth.md`](DECISION_VERIFICATION_extended_depth.md) |
| Deliberation depth | [`DECISION_DELIBERATION.md`](../DECISION_DELIBERATION.md) · [`DECISION_DELIBERATION_extended_depth.md`](DECISION_DELIBERATION_extended_depth.md) |
