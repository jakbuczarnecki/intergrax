## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_embedded_detail.md`](plan/satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


# Experimentation And Developer Experience - Implementation Plan

**Architecture (1:1):** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Last updated:** 2026-06-20 - **P2-ARCH-13** Experimentation/DX architecture vs implementation rules boundary.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_embedded_detail.md`](plan/satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_embedded_detail.md). §6.1 maintenance queues - open P0/P1 only
- **Use** `Read` with offset/limit - open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

<a id="protocol-v22-pba-fix-d--experiment-persistence-port-2026-08-18"></a>

### Protocol v2.2 - PBA-FIX-D - Experiment persistence port (2026-08-18)

**Status:** `ACCEPTED / PLANNED`
**Priority:** P2
**Type:** Arch / Wire / Proof
**Finding:** [`AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-05`](../../audit_results/2026-08-18/PROVIDER_BACKEND_ABSTRACTION.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Outcome (planning only):**

- Introduce/use provider-neutral experiment persistence port.
- Inject it into `ExperimentSession`/debug consumers.
- Retain SQLite as lab/default composition option.
- Prove consumer behavior using the port/substitutable test double.

**Remediation rules:**

- Revalidate finding against then-current `development` HEAD before implementation.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-PROVIDER-BACKEND-ABSTRACTION-PERSIST-1.

**Recommended remediation order (prioritization, not dependency graph):** PBA-FIX-A → PBA-FIX-B → PBA-FIX-C → PBA-FIX-D

---

<a id="protocol-v2-remediation-2026-08-18--accepted--planned"></a>

## Protocol v2 remediation (2026-08-18) - ACCEPTED / PLANNED

**Source:** [`docs/audit_results/2026-08-18/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../../audit_results/2026-08-18/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) · audited_sha `84b2477571650ade894f2d52a6b5398aa86922cc` · verdict **FAIL** · 0 CRITICAL / 5 HIGH / 2 MEDIUM / 0 LOW · operator accepted 2026-08-21

**Status rule:** all rows below are **ACCEPTED / PLANNED** only. Nothing **IMPLEMENTED**, **VERIFIED**, or **CLOSED** in this persistence task. Historical MVP-EVOL / DX-IDEA **Done** rows above are **not** rewritten. **MVP-EVOL.7** route exposure remains a historical delivery fact; **DX-06** owns the residual HTTP functional defect.

### DX-EXPERIMENT-IDENTITY-INTEGRITY - experiment ownership and run linkage

| Field | Value |
|-------|-------|
| **Priority** | P0 / P1 |
| **Owns findings** | AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-01, AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-05 |
| **Intent** | Experiment ownership and run linkage use canonical tenant and execution identity |
| **Cross-links** | [`IDENTITY_TRUST.md`](../../architecture/IDENTITY_TRUST.md) - IDT-FIX-A, ITI-FIX-B |
| **Primary modules** | `intergrax/experiments/models.py`, `store.py`, `workflow.py` |
| **Architecture ref** | [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) - [Protocol v2 experimentation and developer experience target invariants (2026-08-18)](../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md#protocol-v2-experimentation-and-developer-experience-target-invariants-2026-08-18) §1, §5 |
| **Status** | **ACCEPTED / PLANNED** |

### DX-EVALUATION-EVIDENCE-INTEGRITY - criteria and product evidence identity

| Field | Value |
|-------|-------|
| **Priority** | P0 / P1 |
| **Owns findings** | AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-02, AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-03, AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-04 |
| **Intent** | Experiment criteria are real executable semantics; product/satisfaction evidence preserves tenant-scoped identity |
| **Primary modules** | `intergrax/experiments/workflow.py`, `product_kpi_registry.py`, `user_satisfaction.py`, `online_evaluation_models.py`, `online_evaluation_registry.py` |
| **Architecture ref** | same Protocol v2 section §2–§4 |
| **Status** | **ACCEPTED / PLANNED** |

### DX-SURFACE-PERSISTENCE-INTEGRITY - CLI/HTTP service boundary and lab persistence

| Field | Value |
|-------|-------|
| **Priority** | P1 / P2 |
| **Owns findings** | AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-06, AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-07 |
| **Intent** | CLI and HTTP share a real service boundary; lab evidence stores have explicit safe persistence/concurrency semantics |
| **Cross-links** | [PBA-FIX-D](#protocol-v22-pba-fix-d--experiment-persistence-port-2026-08-18) - experiment persistence port; do not duplicate |
| **Primary modules** | `intergrax/cli/mvp_evolution.py`, `mvp_evolution_routes.py`, `product_kpi_registry.py`, `online_evaluation_registry.py` |
| **Architecture ref** | same Protocol v2 section §6–§7 |
| **Status** | **ACCEPTED / PLANNED** |

---

## Architecture doc alignment (P2-ARCH)

| ID | Scope | Status |
|----|-------|--------|
| **P2-ARCH-13** | Clarify Experimentation/DX architecture vs implementation rules boundary | **Done** (2026-06-20) |

---

## Phase DX-IDEA - Idea intake audit (historical)

**Status:** **Done** - superseded by canonical [`docs/audit_results/AUDIT_PROTOCOL.md`](../../../audit_results/AUDIT_PROTOCOL.md) (protocol v2, 2026-08-18). Historical Mode I workflow rows retained for plan traceability only.

| ID | Gap | Priority | Status |
|----|-----|----------|--------|
| DX-IDEA-01 | Mode I indexed in hub and architecture §43.2 surface | P2 | **Done** |
| DX-IDEA-02 | Legacy bootstrap ↔ orchestrator consistency gate (removed with protocol v2) | P3 | **Done** |
| DX-IDEA-03 | Natural-language idea intake workflow | P2 | **Done** |

**Delivery rule:** One **DX-IDEA-\*** ID per PR → update this table → verification green.

**no ADR needed** - documentation and DX workflow only; no runtime contract change.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_appendices.md`](plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_appendices.md) | appendices |
| [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_implementation_history.md`](plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_implementation_history.md) | implementation history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## 5. Definition of Done (Global)



1. **Contract** - Pydantic / Protocol public API

2. **Trace** - state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** - unit + integration, deterministic, no network

4. **Documentation** - update this plan + [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** - `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** - extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** - for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** - hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** - harness phases MUST NOT implicitly include K.1/K.2 or new product hosts

### LCI-0B - LangChain architecture boundary guard

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | APPROVED |
| **Owner** | Experimentation and Developer Experience |
| **Checker** | `scripts/maintenance/check_langchain_boundary.py` |
| **CI** | PR smoke + full governance |
| **Baseline** | `scripts/maintenance/langchain_boundary_grandfather.json` |

### LCI-1D - Knowledge document conformance gate

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | APPROVED |
| **Owner** | Experimentation and Developer Experience |
| **Checker** | `scripts/maintenance/check_knowledge_document_conformance.py` |
| **Tests** | `tests/unit/knowledge/contracts/test_document_conformance.py`; `tests/unit/architecture/test_knowledge_document_conformance_gate.py` |
| **CI** | PR smoke + full governance |
| **Next ownership** | RAG / INTEGRATIONS - LCI-2A |



---

## Phase MVP-EVOL - MVP-to-product evolution layer (Band 2at - planned)

**Status:** **Done** (2026-06-09) - architecture canon §44; MVP-EVOL.1–6 implemented.

**Goal:** Deliver systematic **prototype → MVP → production** tooling: simulation harness, replay UX, KPI/satisfaction hooks, and promotion gate automation - competitive DX for product teams on Intergrax.

**Prerequisites:** Phase DX **Done**; Phase EVAL **Done**; lab host **Done**.

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| MVP-EVOL-DOC.1 | MVP0 | Canon §44 + hub cross-ref | **Done** | `docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | ORCHE §58 index |
| MVP-EVOL.1 | MVP1 | **Promotion gate script** - G0–G2 CI checks (runnable, eval baseline, policy) | **Done** | `scripts/gates/check_mvp_promotion_gates.py` | G0–G2 OK |
| MVP-EVOL.2 | MVP2 | **Agent simulator CLI** - multi-agent failure/contention scenarios | **Done** | `intergrax/cli/mvp_evolution.py` | `intergrax mvp simulate` |
| MVP-EVOL.3 | MVP3 | **Trace replay** - reconstruct from trace store | **Done** | `intergrax/cli/mvp_evolution.py` | `intergrax mvp replay` |
| MVP-EVOL.4 | MVP4 | **Product KPI registry** - tenant-scoped metric definitions + export | **Done** | `product_kpi_registry.py` | unit tests deferred |
| MVP-EVOL.5 | MVP5 | **User satisfaction adapter** - thumbs / CSAT event schema + online eval bridge | **Done** | `user_satisfaction.py` | `test_user_satisfaction.py` |
| MVP-EVOL.6 | MVP6 | **Author guide appendix** - MVP evolution playbook | **Done** | `guides/AGENT_CREATION_GUIDE.md` Appendix X | TOC + scripts table |
| MVP-EVOL.7 | Exposure | **Tier-3 router optional** - HTTP endpoints for simulate/replay/KPI export (or document CLI-only canon) | **Done** | `mvp_evolution_routes.py` · lab `/v1/mvp/*` when `LAB_HARNESS=true` | CLI remains canonical |

**Cross-plan:** MVP-EVOL.2 ↔ ORCH CFG matrix; MVP-EVOL.5 ↔ OBS + EVAL online registry; promotion G4–G5 ↔ Phase V / W-OPS.

**Audit note (2026-06-09):** MVP-EVOL.1–6 **Done**; remaining debt is **exposure** (CLI vs product HTTP) - see [`architecture/ORCHESTRATION.md`](../../architecture/ORCHESTRATION.md) §59.4.

**Protocol v2 note (2026-08-21):** MVP-EVOL.7 **Done** records route **exposure** delivered; residual HTTP functional defect is **DX-06** in [Protocol v2 remediation](#protocol-v2-remediation-2026-08-18--accepted--planned) - not a reopen of MVP-EVOL.7 status.

**Explicitly excluded:** Product analytics SaaS UI; K.1/K.2 feature work.

---
