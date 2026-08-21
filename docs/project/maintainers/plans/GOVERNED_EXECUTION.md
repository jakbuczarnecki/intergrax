# Governed Execution — Implementation Plan

## Ownership / canonical architecture

**Canonical architecture:** [`docs/project/architecture/GOVERNED_EXECUTION.md`](../../architecture/GOVERNED_EXECUTION.md)

- The architecture document owns target design and invariants for Governed Execution.
- This plan owns implementation and remediation work for the Governed Execution domain.
- Audit source for current accepted blocks: [`docs/audit_results/2026-08-18/POLICY_GOVERNANCE.md`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md) (AUDIT-5, audited 2026-08-19).

## Current state

Governed Execution mechanisms already exist in the platform (policy evaluation, meaningful-side-effect contracts, collaborative-work enforcement, HITL continuation). AUDIT-5 identified accepted gaps requiring remediation. **None of the PG-FIX blocks below are implemented or verified** by audit persistence; later parallel development commits must not be treated as closure of these historical findings.

## Accepted remediation blocks

### PG-FIX-A — Canonical side-effect governance spine

**Status:** ACCEPTED / PLANNED

**Findings:**

- [`AUDIT-20260818-POLICY_GOVERNANCE-01`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md)
- [`AUDIT-20260818-POLICY_GOVERNANCE-03`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md)

**Target:**

- One canonical meaningful-side-effect authorization path for production consumers.
- Product adapters may adapt domain data but must not own an independent governance semantics path.
- Effective authorization can compose identity/authority, tenant/workspace, resource, target, effect kind, action, and exact side-effect scope.

**Acceptance criteria:**

- All production meaningful-side-effect consumers use the canonical boundary or a thin adapter into it.
- Duplicate evaluator ownership removed.
- Authority semantics are identical across consumers.
- Conformance tests demonstrate no bypass.

### PG-FIX-B — Safe policy resolution semantics

**Status:** ACCEPTED / PLANNED

**Finding:**

- [`AUDIT-20260818-POLICY_GOVERNANCE-02`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md)

**Target:**

- Explicit deterministic precedence/specificity for meaningful-side-effect policy resolution.
- Broad ALLOW cannot silently shadow a more-specific DENY because of list order.

**Acceptance criteria:**

- Precedence semantics are explicit in contract/tests.
- DENY/specificity behavior is deterministic.
- Ordering mistakes cannot weaken authorization.

### PG-FIX-C — Scoped approval consumption

**Status:** ACCEPTED / PLANNED

**Finding:**

- [`AUDIT-20260818-POLICY_GOVERNANCE-04`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md)

**Target:**

- Canonical `GovernedContinuationApprovalGrant` is consumable by the exact side-effect continuation it authorizes.
- Approval never becomes global ALLOW.
- Exact task/run/operation/resource/scope/policy/pause/request binding is preserved.

**Acceptance criteria:**

- REQUIRE_HUMAN → verified approval → exact continuation path is closed.
- Wrong/stale/mismatched grant fails closed.
- DENY is never overridden merely because approval exists.

**Note:** Later parallel G5C commits may exist on current development. Do not mark this block implemented or verified from those commits. Historical audit remains tied to its audited SHA.

### PG-FIX-D — Explicit policy matching

**Status:** ACCEPTED / PLANNED

**Finding:**

- [`AUDIT-20260818-POLICY_GOVERNANCE-05`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md)

**Target:**

- Critical policy matching uses explicit typed fields.
- Remove hidden runtime action inference from `rule_id` suffixes unless a separately approved migration requirement exists.

**Acceptance criteria:**

- Match semantics are explicit.
- Rule identifiers are identifiers, not hidden dispatch instructions.
- Tests prove clean-cut behavior.

## Dependencies / cross-layer relationships

- **Identity/Trust** — principal/authority correctness for effective side-effect authorization.
- **Unified Execution Runtime** — runtime policy propagation to governed step boundaries.
- **Reliability / HITL** — exact approval grant consumption and continuation correlation.
- **Tools / Integrations** — meaningful-side-effect consumers must route through the canonical spine.

<a id="cla-control-plane-governance-integrity-2026-08-18"></a>

### CLA-CONTROL-PLANE-GOVERNANCE-INTEGRITY — Control-plane mutation governance boundary (Protocol v2 · 2026-08-18)

**Status:** `ACCEPTED / PLANNED`
**Priority:** P0
**Type:** Meta-architecture / governance taxonomy
**Source:** [`AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-04`](../../audit_results/2026-08-18/CROSS_LAYER_ARCHITECTURE.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Target:**

- extend Governed Execution with **CONTROL_PLANE_MUTATION** as a Governance Evaluation Point class
- minimum shared authority context: principal, tenant/scope, resource identity, current/target revision, risk, approval evidence, mutation/idempotency identity
- domain owners still execute their own mutations — no universal mutation executor or `GovernanceEngine`

**Consumers (cross-link only):**

- [`AGENT_DISTRIBUTION` plan](AGENT_DISTRIBUTION.md) — activation/rollback
- [`ADAPTIVE_HARNESS_INTELLIGENCE` plan](ADAPTIVE_HARNESS_INTELLIGENCE.md) — apply/rollback
- [`ELASTIC_CAPACITY_AND_SCALING` plan](ELASTIC_CAPACITY_AND_SCALING.md) — **ECP-GOVERNED-ACTION-INTEGRITY**
- [`NEXUS_EXECUTION_FLOW` plan](NEXUS_EXECUTION_FLOW.md) / [`TIER3_APPLICATION_ENVIRONMENT` plan](TIER3_APPLICATION_ENVIRONMENT.md) — **E2E-CONTROL-AUTHORITY-INTEGRITY**
- [`PLATFORM_PLUGINS` plan](PLATFORM_PLUGINS.md) — activation/admission

**Acceptance criteria:**

- G3B coverage table marks live **CONTROL_PLANE_MUTATION** status honestly until consumers converge
- shared boundary documented; specialized executors unchanged in ownership
- conformance demonstrates no second control-plane permission engine

**Remediation rules:**

- **Not implemented** by audit persistence task AUDIT-20260818-CROSS-LAYER-ARCHITECTURE-PERSIST.

## Verification expectations

Implementation requires code, tests, and independent verification before any block moves beyond **ACCEPTED / PLANNED**. Audit persistence does not constitute implementation or verification evidence.

---
