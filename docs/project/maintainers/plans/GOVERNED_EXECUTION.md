# Governed Execution - Implementation Plan

## Ownership / canonical architecture

**Canonical architecture:** [`docs/project/architecture/GOVERNED_EXECUTION.md`](../../architecture/GOVERNED_EXECUTION.md)

- The architecture document owns target design and invariants for Governed Execution.
- This plan owns implementation and remediation work for the Governed Execution domain.
- **UE-DOC-0.7 (2026-08-26):** architecture hub frozen Execution-centric governance, admission vs inner evaluation points, authority inheritance, guardrails terminology, and **UEA-INV-021** no-bypass invariant - plan rows unchanged; future implementation must not reintroduce executor-local governance bypass.
- Audit source for current accepted blocks: [`docs/audit_results/2026-08-18/POLICY_GOVERNANCE.md`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md) (AUDIT-5, audited 2026-08-19).

## Current state

Governed Execution mechanisms already exist in the platform (policy evaluation, meaningful-side-effect contracts, collaborative-work enforcement, HITL continuation). AUDIT-5 identified accepted gaps requiring remediation.

**GOV-FINAL-1 (2026-09-17):** Maintainer status synchronized to `development` code truth. **Architecture authority:** [`architecture/GOVERNED_EXECUTION.md`](../../architecture/GOVERNED_EXECUTION.md) only — this plan is roadmap/status, not a second SSOT. Gap ledger: [`qualification/GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md`](../qualification/GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md).

**GOV-FINAL-2 (session):** GR-3 `authorize_and_execute` allowlist includes Decision-bound Execution adapter; GR-4 removes undocumented Nexus import from `meaningful_side_effect_authorization.py` (pause via governed-continuation bridge). Enterprise qualification matrix still open (GR-8/10/13).

**PG-FIX legacy map (final reconciliation — auditable):**

| Block | Original problem | Superseding GR-* | Status after GOV-FINAL-1 | Remaining gap |
| ----- | ---------------- | ---------------- | ------------------------ | ------------- |
| PG-FIX-A | Duplicate / adapter-owned side-effect policy paths | GR-3 inner spine | **SUPERSEDED_BY_GR_3** (mechanism) — **PARTIAL** qualification | Not every consumer path wired; universal coverage unproven |
| PG-FIX-B | Non-deterministic policy precedence | GR-4 policy core | **SUPERSEDED_BY_GR_4** — **IMPLEMENTED** / enterprise sign-off **OPEN** (GR-13) | GR-4-R1 Nexus-neutral bundle assembly |
| PG-FIX-C | Scoped grant vs global ALLOW | GR-1 identity + GR-5 continuation | Grant binding **CLOSED** (GR-1); lifecycle **PARTIAL** (GR-5 candidate) | Strategy-wide HITL qual (GR-10) |
| PG-FIX-D | Hidden `rule_id` suffix matching | GR-4 policy core | **SUPERSEDED_BY_GR_4** — **IMPLEMENTED** / enterprise sign-off **OPEN** (GR-13) | Plugin enterprise certification (GR-11) |

Historical AUDIT-5 rows remain **context**; they are not erased. **Enterprise CLOSED** for PG-FIX blocks requires GR-13 / GR-16 — not claimed here.

**Active roadmap:** GR-1…GR-7 implementation slices landed on `development`; **GR-8+** and enterprise qualification remain open. See gap ledger § GR enterprise roadmap and architecture § Governance implementation truth.

Audit persistence alone never constitutes implementation or verification evidence.

## Accepted remediation blocks

### PG-FIX-A - Canonical side-effect governance spine

**Status:** IMPLEMENTED (core) — VERIFIED partial — **not CLOSED** (GR-0; see gap ledger GOV-GAP-005, GOV-REBASE-02)

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

### PG-FIX-B - Safe policy resolution semantics

**Status:** IMPLEMENTED — VERIFIED partial (`tests/unit/runtime/policy/test_pg_fix_b_side_effect_policy_precedence.py`) — **not CLOSED** (GR-4 requalification)

**Finding:**

- [`AUDIT-20260818-POLICY_GOVERNANCE-02`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md)

**Target:**

- Explicit deterministic precedence/specificity for meaningful-side-effect policy resolution.
- Broad ALLOW cannot silently shadow a more-specific DENY because of list order.

**Acceptance criteria:**

- Precedence semantics are explicit in contract/tests.
- DENY/specificity behavior is deterministic.
- Ordering mistakes cannot weaken authorization.

### PG-FIX-C - Scoped approval consumption

**Status:** IMPLEMENTED (mechanism) — VERIFIED partial (G5C-2B / PG-FIX-C tests) — **not CLOSED** — grant/side-effect identity **CLOSED** (GR-1); UER pause ownership **OPEN** (GR-5)

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

**Note:** G5C commits implemented the historical mechanism; GR-0 confirms mechanism soundness separate from Attempt/Execution binding. AUDIT-5 SHA context preserved in [`POLICY_GOVERNANCE.md`](../../audit_results/2026-08-18/POLICY_GOVERNANCE.md).

### PG-FIX-D - Explicit policy matching

**Status:** IMPLEMENTED — VERIFIED partial (`test_pg_fix_d_explicit_policy_action_matching.py`) — **not CLOSED** (GR-4)

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

- **Identity/Trust** - principal/authority correctness for effective side-effect authorization.
- **Unified Execution Runtime** - runtime policy propagation to governed step boundaries.
- **Reliability / HITL** - exact approval grant consumption and continuation correlation.
- **Tools / Integrations** - meaningful-side-effect consumers must route through the canonical spine.

<a id="cla-control-plane-governance-integrity-2026-08-18"></a>

### CLA-CONTROL-PLANE-GOVERNANCE-INTEGRITY - Control-plane mutation governance boundary (Protocol v2 · 2026-08-18)

**Status:** `ACCEPTED / PLANNED`
**Priority:** P0
**Type:** Meta-architecture / governance taxonomy
**Source:** [`AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-04`](../../audit_results/2026-08-18/CROSS_LAYER_ARCHITECTURE.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Target:**

- extend Governed Execution with **CONTROL_PLANE_MUTATION** as a Governance Evaluation Point class
- minimum shared authority context: principal, tenant/scope, resource identity, current/target revision, risk, approval evidence, mutation/idempotency identity
- domain owners still execute their own mutations - no universal mutation executor or `GovernanceEngine`

**Consumers (cross-link only):**

- [`AGENT_DISTRIBUTION` plan](AGENT_DISTRIBUTION.md) - activation/rollback
- [`ADAPTIVE_HARNESS_INTELLIGENCE` plan](ADAPTIVE_HARNESS_INTELLIGENCE.md) - apply/rollback
- [`ELASTIC_CAPACITY_AND_SCALING` plan](ELASTIC_CAPACITY_AND_SCALING.md) - **ECP-GOVERNED-ACTION-INTEGRITY**
- [`NEXUS_EXECUTION_FLOW` plan](NEXUS_EXECUTION_FLOW.md) / [`TIER3_APPLICATION_ENVIRONMENT` plan](TIER3_APPLICATION_ENVIRONMENT.md) - **E2E-CONTROL-AUTHORITY-INTEGRITY**
- [`PLATFORM_PLUGINS` plan](PLATFORM_PLUGINS.md) - activation/admission

**Acceptance criteria:**

- G3B coverage table marks live **CONTROL_PLANE_MUTATION** status honestly until consumers converge
- shared boundary documented; specialized executors unchanged in ownership
- conformance demonstrates no second control-plane permission engine

**Remediation rules:**

- **Not implemented** by audit persistence task AUDIT-20260818-CROSS-LAYER-ARCHITECTURE-PERSIST.

## Verification expectations

Implementation requires code, tests, and independent verification before any block moves to **CLOSED**. Audit persistence does not constitute implementation or verification evidence.

## GR enterprise roadmap (GOV-FINAL-1 code truth)

Full evidence rows and G-stage disposition: [`GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md`](../qualification/GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md). Status vocabulary: mechanism may be **IMPLEMENTED** while GR scope or enterprise qualification remains **OPEN**.

| ID | Task | Status |
| --- | --- | --- |
| GR-0 | Architecture Rebase & Gap Ledger | **CLOSED** |
| GR-1 | Execution Identity Rebinding | **CLOSED** |
| GR-2 | Execution Admission Governance | **IMPLEMENTED** — qualification **OPEN** (candidate closed; audit pending) |
| GR-3 | Inner Evaluation Spine Reconciliation | **IMPLEMENTED** — qualification **OPEN** (GR-3-R1/R2 done) |
| GR-4 | Policy Resolution & Catalog Requalification | **IMPLEMENTED** — qualification **OPEN**; GR-4-R1 **OPEN** |
| GR-5-ADR1 | Canonical Execution HITL Continuation Ownership | **CLOSED** — [ADR-GR-5-001](../../technical/adr/entries/2026-09-15/ADR-GR-5-001.md) |
| GR-5-R1 | Canonical Execution Continuation Contract | **CLOSED** |
| GR-5-R2–R5 | Pause/resume integration, projection, restart qual | **IMPLEMENTED** — qualification **OPEN** (candidate closed slices) |
| GR-5 | HITL / Governed Continuation Rebase | **IMPLEMENTED** — enterprise qualification **OPEN** (GR-10) |
| GR-6 | Decision → Governance Integration | **IMPLEMENTED** — qualification **OPEN** (host-qualified; not all strategies) |
| GR-7 | External Effect / Reliability Boundary | **IMPLEMENTED** — qualification **OPEN** (ERL path; Reliability ≠ Governance) |
| GR-8 | Governance Evidence Integration | **CANDIDATE CLOSED — PUBLIC CONTRACT FROZEN** — [ADR-GR-8-001](../../technical/adr/entries/2026-09-17/ADR-GR-8-001.md); independent final audit pending |
| GR-9 | Diagnostic Consumption Proof | **OPEN** |
| GR-10 | Execution Strategy Coverage | **FINAL CLOSED within formally defined GR-10 scope** — full per-GEP Governance Evidence **deferred to GR-13** |
| GR-10-FINAL | Strategy governance requalification | **CLOSED (qualification artifact)** — independent GitHub audit before GR-10 slice CLOSED |
| GR-10-R3-R1 | UAEP kernel authority internal carrier | **CANDIDATE CLOSED — PUBLIC AGENT BOUNDARY RESTORED** — `UaepKernelStepExecution` adapter carrier; independent GitHub audit required |
| GR-10-R3 | UAEP AGENT_DECISION authority on kernel path | **CANDIDATE CLOSED — FUNCTIONAL + ARCHITECTURAL CONFORMANCE COMPLETE** — kernel `policy_pre` via internal carrier → `GovernanceResolution`; G3B deny proof green; independent GitHub audit required |
| GR-10-R2-ADR1 | PRE_MODEL identity + intake boundary (ADR) | **CANDIDATE CLOSED** — [ADR-GR-10-001](../../technical/adr/entries/2026-09-18/ADR-GR-10-001.md) accepted; independent GitHub audit before CLOSED |
| GR-10-R2-ADR1-R1 | PRE_MODEL identity SSOT reconciliation | **CANDIDATE CLOSED** — current docs reconciled to GR-10-R1 semantics; independent GitHub audit before CLOSED |
| GR-10-R2-C1 | PRE_MODEL public contract revision | **CANDIDATE CLOSED** — `principal_id` required on `evaluate_pre_llm`; independent GitHub audit before CLOSED |
| GR-10-R2-R1 | PRE_MODEL runtime identity conformance | **CANDIDATE CLOSED — RUNTIME IDENTITY CONFORMANCE COMPLETE** — independent GitHub audit before CLOSED |
| GR-10-R2 | INFERENCE PRE_MODEL runtime conformance | **CANDIDATE CLOSED** — [ADR-GR-10-001](../../technical/adr/entries/2026-09-18/ADR-GR-10-001.md); independent audit after **GR-10-R2-R1** |
| GR-10-R4 | INFERENCE root admission enterprise qualification | **CANDIDATE CLOSED — INFERENCE ROOT ADMISSION NOT_APPLICABLE BY ARCHITECTURE** — no independent production root; `tests/qualification/governance/strategy/test_gr10_r4_inference_root_admission_qualification.py`; independent GitHub audit required |
| GR-10-R5 | INFERENCE Inner Governance enterprise qualification | **CANDIDATE CLOSED — INFERENCE INNER GOVERNANCE NOT_APPLICABLE BY ARCHITECTURE** — GR-3/MSE spine row distinct from PRE_MODEL Policy evaluation; `test_gr10_r5_inference_inner_governance_qualification.py`; independent GitHub audit required |
| GR-10-R6 / R6-R1 | INFERENCE Governance Evidence | **CANDIDATE CLOSED** — mandatory composition + PRE_MODEL verdict semantics; independent GitHub audit required |
| GR-10-R7 | AGENTIC & ORCHESTRATION residual strategy requalification | **CANDIDATE CLOSED — MATRIX REQUALIFIED** — `GR10_AGENTIC/ORCHESTRATION_CAPABILITY_SEMANTICS` + `test_gr10_r7_agentic_orchestration_residual_requalification.py`; independent GitHub audit required |
| GR-10-A1 / A1-R1 | AGENTIC production scope & ACP reachability | **CLOSED** — P-UAEP qualified; P-ACP reachability documented |
| GR-10-A2 | AGENTIC execution model architecture decision | **CLOSED** — [ADR-GR-10-004](../../technical/adr/entries/2026-09-21/ADR-GR-10-004-agentic-execution-model-uaep-canonical.md) `UAEP_CANONICAL_ACP_EXPLICIT`; independent GitHub audit required |
| GR-10-A3 / A3-R1 | UAEP-canonical decoupling + checkpoint authority | **CLOSED** — checkpoint ≠ `acp.session.v1`; P-UAEP `RuntimeCheckpoint` / `TaskCheckpointPersistence` |
| GR-10-A4 / A4-R1 | Qualification blocker + acceptance evidence integrity | **CLOSED** — g3b import gates; 05c/05d/05e acceptance |
| GR-10-A5 | R7 final AGENTIC qualification reconciliation | **CLOSED** — matrix aligned to `GR10_AGENTIC_CAPABILITY_SEMANTICS` |
| GR-10 AGENTIC Final Recertification & Closure | Whole GR-10 formal closure | **FINAL CLOSED** — `GR10_OVERALL_FORMAL_CLOSURE` SSOT within formally defined GR-10 scope |
| GR-12-A1 | Control-plane inventory + architecture | **CLOSED** |
| GR-12-A2 | Mandatory CLA-04 composition | **FINAL CLOSED** |
| GR-12-A3 | Core control-plane qualification | **FINAL CLOSED** |
| GR-12-A4 | Residual path classification | **CLOSED** |
| GR-12 Architecture Checkpoint | Scope + ownership revalidation | **CLOSED** |
| GR-12-DOC-R1 | Canonical docs/status reconciliation | **CLOSED** |
| GR-12-A4-R1-R1 | Catalog revision + ABA + explicit operator identity | **CLOSED** |
| GR-12-A4-R1-R1-R1 | Catalog atomic registration + qualification closure | **CLOSED** |
| GR-12-A4-R2 | Vector administration governance ADR | **NEXT** |
| GR-12-A4-R3 | Specialized memory governance ADR | **AFTER VECTOR** |
| GR-12 Final Qualification | Final control-plane certification | **AFTER RESIDUALS** |
| GR-10-R9-ADR1 | Canonical orchestration MSE authority contract | **CLOSED — CANONICAL MSE AUTHORITY CONTRACT APPROVED** — [ADR-GR-10-002](../../technical/adr/entries/2026-09-19/ADR-GR-10-002.md); `test_gr10_r9_adr1_mse_authority_contract.py`; independent GitHub audit required |
| GR-10-R9 | ORCHESTRATION MSE production coverage | **CLOSED within GR-10 scope** — R9-R1/R2/R3 qualification slices |
| GR-11 | Plugin & Enterprise Extensibility Certification | **OPEN** (after GR-12) |
| GR-12 | Control-Plane Governance | **IN PROGRESS** — shared CLA-04 spine + core path qualification; residual qualification open (not **CLOSED**) |
| GR-13 | Full Governance Proof Matrix | **OPEN** (after GR-11) — includes `GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED` per [ADR-GR-10-003](../../technical/adr/entries/2026-09-21/ADR-GR-10-003-gr10-gr13-governance-evidence-certification-scope.md) (GR-10-R15-R1); evidence scope **not** folded into GR-12 |
| GR-14 | Real Application Integration — LKW | **OPEN** |
| GR-15 | Governance UX / Application Contract | **OPEN** |
| GR-16 | Enterprise Qualification & Claims | **OPEN** |

---
