<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Decision / Approval / Governance — Multiplayer integration (MP-4 / MP-4R)

**Status:** **MP-4 — FORMALLY CLOSED** · **MP-4R0…MP-4R8 CLOSED** · legacy **MP-4A** `SUPERSEDED_BY_MP4R0` · **MP-4B** `RETIRED` (MP-4R1) · **MP-4C** `RETIRED` (MP-4R2) · **MP-4D** `RETIRED` (MP-4R2) · legacy MP-4E…MP-4H **cancelled/replaced** by MP-4R1…MP-4R8 · **MP-4D1 — CLOSED** · **MP-4D2 — NEXT**
**ADR:** [ADR-MP-009](../technical/adr/entries/2026-09-15/ADR-MP-009.md) (authoritative after MP-4R0) · [ADR-MP-005](../technical/adr/entries/2026-09-08/ADR-MP-005.md) (MP-4A historical; ownership table superseded)
**Feature coordination:** [`MULTIPLAYER_AI`](../capabilities/architecture/MULTIPLAYER_AI.md) · [`COLLABORATIVE_WORK`](COLLABORATIVE_WORK.md)
**Plan (1:1):** [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md)

---

## MP-4 STATUS (canonical)

```text
MP-4 STATUS: FORMALLY CLOSED

R0–R8: CLOSED
Implementation: ENTERPRISE
Architecture: ENTERPRISE
Authority / Security: ENTERPRISE
E2E qualification: CLOSED
Documentation certification: MP-4D1–D8
```

**MP-4D1–D8** are **documentation and proof-closure stages only**; they **do not reopen** MP-4 implementation.

---

## Purpose

Define how **Multiplayer** integrates with canonical platform authorities for Decision, human authorization (Governance/HITL), Execution continuation, Evidence, and Diagnostics — **without** creating parallel lifecycle or truth sources.

MP-4R0 performed ownership rebase, legacy inventory, supersession, and architecture gates. **MP-4R1** retired legacy MP-4B `intergrax/contracts/decision.py` and the dynamic package bridge; `intergrax/contracts/decision/` is **Decision Integration Boundary** only. **MP-4R2** retired legacy MP-4C/D (`intergrax/contracts/approval.py`, `intergrax/approval/**`) after caller proof — canonical human judgment is `decision_human_review` + Governance/HITL only. **MP-4R3** adopts the execution continuation boundary: Multiplayer does **not** own pause/wait/resume lifecycle; public integration is **`ExecutionContinuationPort`** only; Governance/HITL owns authorization; Execution Engine owns lifecycle; Nexus remains **internal** orchestration only. No Multiplayer continuation store was added — `collaborative_work` currently holds provenance references only (no production continuation port caller).

---

## Canonical ownership (frozen at MP-4R0)

| Capability | Canonical owner | Multiplayer role |
|------------|-----------------|------------------|
| Decision identity / version | Decision System (`decision_identity`, …) | References / bindings only (MP-4R4+) |
| Decision lifecycle / resolution / finalization | Decision System + Execution host | None — no second lifecycle |
| Decision human review semantics | Decision System + canonical HITL contracts | Bridge via platform contracts only |
| Authorization / `REQUIRE_HUMAN` | Governance / HITL | Consume — do not re-own |
| Human authorization evidence | Governance / HITL + Evidence contracts | Link — do not own facts |
| Pause / wait / resume | Execution Engine | None |
| Public continuation boundary | `ExecutionContinuationPort` | Reference/display only until integration caller; **no** side-channel resume (MP-4R3) |
| Execution identity / lifecycle | Execution Engine | `ExecutionProvenanceRef` references only |
| Orchestration | Nexus **internal** to Execution Engine | **No public Nexus dependency** |
| Principal / membership / delegation | MP-1 (Collaborative Work) | Reuse |
| WorkItem / Assignment | Multiplayer Collaborative Work (MP-2) | Owner |
| WorkArtifact / WorkArtifactVersion | Multiplayer Collaborative Work (MP-3) | Owner |
| Evidence facts | Observability / Evidence Plane | Emit/link canonical facts (MP-4R5) |
| Historical factual reconstruction | Shared Evidence Plane reconstruction | None |
| Diagnostic interpretation | Central Diagnostics | Read/integration via canonical contracts |
| Multiplayer Decision integration | Multiplayer plane | **Collaborative binding / projection only** |

**Hard invariants:**

```text
Multiplayer MUST NOT own a second Decision lifecycle.
Multiplayer MUST NOT own a second Approval/HITL authority.
Multiplayer MUST NOT own Execution lifecycle.
Multiplayer MUST NOT expose Nexus.
Multiplayer MUST NOT own evidence truth.
Multiplayer MUST NOT own diagnostic interpretation.
```

---

## Target integration model

```text
Collaborative Work (WorkItem / Assignment / WorkArtifact)
        ↓
CollaborativeDecisionBinding (association only)
        ↓
exact DecisionProposalRef
        ↓
Decision System (lifecycle / resolution)
        ↓
Governance (ALLOW / DENY / REQUIRE_HUMAN)
        ↓
Human Review / HITL when required
        ↓
post-human Governance
        ↓
DecisionExecutionAuthorization + current-policy validation
        ↓
ExecutionContinuationPort (pause / resume)
        ↓
Execution Engine (identity + lifecycle; Nexus internal)
        ↓
Evidence Plane (facts)
        ↓
Factual reconstruction
        ↓
Diagnostics (interpretation)
```

---

## MP-4R8 ownership matrix (enterprise closure)

| Concern | Canonical owner | Canonical contract surface |
|---------|-----------------|----------------------------|
| Decision truth | Decision System | `decision_identity`, `decision_lifecycle`, `decision_record`, … |
| Human approval | Human Review / HITL | `decision_human_review`, `HumanApproverEvidence` |
| Governance authorization | Governance | `DecisionGovernanceDecision`, evaluator plugins |
| Execution authorization | Governance-derived authorization | `DecisionExecutionAuthorization`, `mint_validated_execution_authorization` |
| Continuation lifecycle | Execution | `ExecutionContinuationPort` |
| Collaborative work | Multiplayer Collaborative Work | WorkItem, Assignment, WorkArtifact (MP-2/MP-3) |
| Collaborative binding / association truth | Multiplayer Collaborative Work | `CollaborativeDecisionBinding`, `CollaborativeDecisionBindingRepository` |
| Evidence facts (operation execution) | Evidence Plane | `FunctionalEvidencePersistence`, `PlatformFunctionalEvidence` |
| Diagnostics interpretation | Diagnostics | functional diagnostic specs / analyzers |

**Evidence semantics (authoritative):** `CollaborativeDecisionBindingRepository` is the **source of association truth** (WorkItem ↔ `DecisionProposalRef`). The **Evidence Plane** records **operation execution facts** (for example binding-create `OPERATION_OUTCOME`). Association truth is **not** reconstructed from Evidence Plane facts.

**Known non-blocking platform limitation (frozen):** `CollaborativeDecisionBinding` WorkItem ↔ `DecisionProposalRef` association itself is **not** represented by a dedicated frozen Evidence Plane v2 fact kind; MP-4R5 adopts **operation outcome** evidence only — no `OUTPUT_RELATION` / `ARTIFACT_LINEAGE` workaround. **Not** an open MP-4 implementation blocker; future track = Evidence Plane architecture extension.

**R7 E2E qualification (closed):** qualifies the success path **Collaborative Work → Decision → Governance `REQUIRE_HUMAN` → Human Review → post-human Governance → `DecisionExecutionAuthorization` → current-policy validation → `ExecutionContinuationPort` → protected operation → Evidence → Diagnostics**.

**R7 provider scope (precise):** **R7** = **architectural / cross-domain E2E** using **canonical production contracts** and test composition where configured — **not** a claim of full production-deployment E2E on every real provider stack. **Collaborative decision binding PostgreSQL** was **separately real-provider qualified** in **MP-4R4** (real PostgreSQL, cross-process concurrency, idempotency, conflict, semantic dedup).

---

## Legacy MP-4 inventory (post MP-4R1)

| Component | Location | Classification |
|-----------|----------|----------------|
| MP-4B Decision contracts (module) | `intergrax/contracts/decision.py` | **REMOVED** (MP-4R1) — canonical `decision_identity` / DS-CORE only |
| Decision Integration namespace | `intergrax/contracts/decision/__init__.py` | **KEEP** — re-exports integration SPI only; no dynamic legacy bridge |
| Decision Integration Boundary | `intergrax/contracts/decision/integration/**` | **KEEP** — adapter SPI; lifecycle mapping uses `decision_lifecycle` |
| Integration composition root | `intergrax/runtime/decision_integration_composition.py` | `KEEP` — composition boundary for providers/adapters |
| Decision plugin composition | `intergrax/runtime/decision_plugin_composition.py` | `KEEP` — wires canonical decision flow + integration engine |
| MP-4C Approval contracts | `intergrax/contracts/approval.py` | **REMOVED** (MP-4R2) — use `decision_human_review` + Governance/HITL |
| MP-4D Approval service | `intergrax/approval/` | **REMOVED** (MP-4R2) — no second human-review/HITL runtime |
| Legacy approval tests | `test_approval_*`, `tests/unit/approval/*` | **REMOVED** (MP-4R2) |
| Architecture gates | `test_mp4r0_*`, `test_mp4r1_*`, `test_mp4r2_human_review_approval_convergence_gates.py` | `KEEP` |
| MP-4R0 collaborative gates | `tests/unit/runtime/architecture/test_mp4r0_multiplayer_rebase_architecture_gates.py` | `KEEP` — protect Multiplayer production roots |

**MP-4R2 caller proof (production):** zero imports of `intergrax.contracts.approval` or `intergrax/approval/**` outside removed surfaces. Human judgment for Decisions uses `DecisionProposalRef` via `decision_human_review`; workflow states (`ASSIGNED`, `IN_REVIEW`, …) were legacy Approval metadata, not retained as platform authority.

**Decision Integration Boundary — pluginability:** engine depends on `DecisionSystemIntegrationAdapter` / provider protocols; concrete adapters selected at `decision_integration_composition.py` / `decision_plugin_composition.py`; unknown provider fails closed via admission/composition policy; contracts carry no Nexus/vendor types.

**Unique collaborative capability audit:** collaborative association remains a **binding** concern (MP-4R4), not Decision lifecycle ownership.

---

## Anti-substitution rules (retained — canonical owners)

| Forbidden | Correct model |
|-----------|---------------|
| Decision ≡ WorkArtifact / WorkItem state (`Decision != Artifact`) | Canonical Decision + optional WorkItem binding |
| Human Review ≡ Governance ALLOW (`Human Review != Governance ALLOW`) | Human review evidence via `decision_human_review`; governance outcomes separate |
| Governance ALLOW ≡ execution without authorization validation | `DecisionExecutionAuthorization` + current-policy validation before continuation |
| Evidence / Diagnostics as authority (`Evidence != authority`; `Diagnostics != authority`) | Evidence = factual evidence; Diagnostics = interpretation only |
| Binding ≡ Decision (`Binding != Decision`) | Binding associates; Decision System owns lifecycle truth |
| Decision encoded as WorkArtifact body or WorkItem state | Canonical Decision + optional WorkItem binding |
| `WorkArtifactVersion.status = APPROVED` | Human review / governance outcomes via canonical contracts |
| `ExecutionState` substituting approval workflow | Governance/HITL + `ExecutionContinuationPort` |
| Decision outcome substituting TaskState / RunState | `ExecutionProvenanceRef` when correlated |
| Governance evidence substituting ProofReceipt | Evidence Plane owns facts; linkage only |
| Approval/evidence alone authorizing execution | MP-INV-23 — Governed Execution path |

**Normative:** MP-INV-09 (Decision ≠ HITL), MP-INV-23 (approval/evidence ≠ execution authorization).

---

## Contract-first integration (frozen)

```text
Multiplayer
   ↓
stable platform contract / port
   ↓
configured platform implementation
```

**Forbidden for new Multiplayer production modules:**

- `intergrax.runtime.nexus.*`, `GraphExecutor`, `NexusLoop`, `NexusIntakeRunner`
- New `DecisionId` / `DecisionLifecycle*` / `DecisionRuntime` / `MultiplayerDecisionEngine` (outside legacy quarantine)
- `MultiplayerHitlEngine`, `ApprovalRuntime`, `HumanReviewRuntime`
- `EvidenceStore`, `ExecutionReconstructor`, `DiagnosticEngine`, `ProblemLifecycleEngine` as Multiplayer-owned authorities

**Reuse (mandatory):** `ExecutionContinuationPort`, Decision System contracts, Governance/HITL contracts, Evidence Plane contracts, Diagnostic read surfaces, MP-1 authority.

---

## MP-4R roadmap (replaces MP-4E…MP-4H)

| Slice | Status |
|-------|--------|
| **MP-4R0** — Core rebase & supersession gate | **CLOSED** |
| **MP-4R1** — Decision contract convergence | **CLOSED** |
| **MP-4R2** — Human review / Approval convergence | **CLOSED** |
| MP-4R3 — Execution continuation integration | **CLOSED** |
| MP-4R4 — Collaborative decision binding | **CLOSED** |
| MP-4R5 — Evidence Plane adoption | **CLOSED** — operation-outcome evidence; binding association evidence gap **deferred** (frozen Evidence Plane contract) |
| MP-4R6 — Legacy removal & migration | **CLOSED** — retired MP-4 authority paths; fail-closed legacy human-decision provenance; admin-only disposition CLI |
| MP-4R7 — Enterprise integration qualification | **CLOSED** — `testing_support/mp4r7_enterprise_integration/` + architecture gates + E2E qualification tests |
| MP-4R8 — Final closure audit | **CLOSED** — meta-gates `test_mp4r8_final_closure_meta_gates.py`; formal MP-4 program **FORMALLY CLOSED** |

Detail: [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md).

---

## Enterprise Documentation & Proof Closure (MP-4D)

**Does not reopen MP-4 implementation.** Tracks documentation and certification alignment only.

| Stage | Status | Purpose |
|-------|--------|---------|
| **MP-4D1** | **CLOSED** | Synchronize documentation state with actual closed implementation |
| **MP-4D2** | **NEXT** | Consolidate canonical architecture documentation into one coherent entry point |
| MP-4D3 | NOT STARTED | Add professional visual architecture layer (Mermaid/SVG); **D3 owns** visual architecture closure |
| MP-4D4 | NOT STARTED | Create E2E proof / invariant-to-test matrix |
| MP-4D5 | NOT STARTED | Document provider/persistence qualification boundaries |
| MP-4D6 | NOT STARTED | Certify contract-first/pluginability boundaries in docs |
| MP-4D7 | NOT STARTED | Add documentation regression protection |
| MP-4D8 | NOT STARTED | Final enterprise documentation audit |

Plan tracking: [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md) · capability summary: [`MULTIPLAYER_AI`](../capabilities/plan/MULTIPLAYER_AI.md).

---

## Legacy slice status (historical)

| Stary slice | Status |
|-------------|--------|
| MP-4A | `SUPERSEDED_BY_MP4R0` |
| MP-4B | `RETIRED` (MP-4R1) |
| MP-4C | `RETIRED` (MP-4R2) |
| MP-4D | `RETIRED` (MP-4R2) |
| MP-4E | `CANCELLED_BEFORE_START` |
| MP-4F | `CANCELLED_REPLACED_BY_EVIDENCE_ADOPTION` |
| MP-4G | `CANCELLED_IN_OLD_FORM` |
| MP-4H | `REPLACED_BY_MP4R8` |

---

## Related documents

| Document | Role |
|----------|------|
| [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md) | Canonical Decision authority |
| [`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md) | Governance / HITL |
| [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) | Execution + internal Nexus |
| [`OBSERVABILITY.md`](OBSERVABILITY.md) | Evidence Plane |
| [`DIAGNOSTICS.md`](DIAGNOSTICS.md) | Diagnostic interpretation |
| [`COLLABORATIVE_WORK.md`](COLLABORATIVE_WORK.md) | MP-1…MP-3 ownership |
