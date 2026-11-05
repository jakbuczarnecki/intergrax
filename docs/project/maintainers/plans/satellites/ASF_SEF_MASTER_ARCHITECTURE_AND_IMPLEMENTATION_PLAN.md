# ASF/SEF — Master architecture & implementation plan (published narrative)

**Status:** Architecture narrative — **Proposed** (P1-B1 publication; not product implementation)  
**Binding SSOT:** [`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md`](../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md) (38 REQ · 82 WBS · 30 GAP · 52 CAP · 30 contracts — no competing traceability)  
**Operator:** OD-SEF-01 **APPROVED** (Variant **C + D-lite**) · Ticket [`T-IGX-SEF-A`](../../../../governance/tickets/T-IGX-SEF-A.md)

## Purpose

End-to-end **agentic software/scenario engineering factory**: business intent → bounded engineering → evidence → PR/artifact — **distinct** from scenario production admission (`SCENARIO-GATE`, PROD-Q Dimension B).

## Non-goals

Second Nexus, second ToolRuntime, second FRZ/ECAP SSOT, implementer B6, Cursor-as-runtime-authority, Option 4 pilot path.

## Two planes (OD-SEF-01)

| Plane | Authority | Canonical components |
|-------|-----------|----------------------|
| **Engineering / Delivery** | Engineering Execution Authority (EEA) | DevOS (`verify_plan_approval`, `igx_gov ticket`), engineering orchestrator (PROPOSED `agents/sef_orchestration/`), `CodingExecutionBackend`, workspace lease, bounded repair |
| **Scenario runtime** | Host admission + Nexus | `HostTaskExecutionPort` → `NexusLoop` → `PolicyEngine` → `RuntimeToolInvoker` / ToolRuntime |

Engineering produces Tier-2/3 artifacts and evidence; scenario runtime **runs admitted** scenarios only.

## Critical path (roadmap order preserved)

EVOL/ENT formal B6 → **ARCH-FREEZE** → ADR-SEF-001 **Proposed** (P1) → engineering contracts → REF-A/B/C → GATE-SEF-FULL → dossier → scoped B6 → **SEF-X ∥ SCENARIO-READINESS-X** (Option 3).

## DevOS ↔ engineering job FSM

Ticket lifecycle (`governance/.../task_lifecycle.yaml`) ≠ engineering job FSM — sync spec **T-P3-001**. Engineering jobs start only after **`plan_approved`** + mechanical `verify_plan_approval.py`.

## Boundaries (summary)

| Operation | Engineering | Scenario runtime |
|-----------|-------------|------------------|
| Repo edit via `CodingExecutionBackend` | ALLOWED (lease) | DENY Nexus shortcut |
| Local pytest / qual | ALLOWED | ALLOWED in harness/proof hosts |
| Production external tool effects | DENY | ALLOWED if admitted + PROD-Q |
| REF-A/B/C runtime demo | proof host + HostTaskExecutionPort | YES |

## Related published artifacts

| Artifact | Path |
|----------|------|
| Semantic ownership matrix | [`ASF_SEF_SEMANTIC_OWNERSHIP_AND_REUSE_MATRIX.md`](ASF_SEF_SEMANTIC_OWNERSHIP_AND_REUSE_MATRIX.md) |
| Traceability index | [`ASF_SEF_TRACEABILITY_INDEX.md`](ASF_SEF_TRACEABILITY_INDEX.md) |
| Operator decisions | [`../../qualification/sef/ASF_SEF_OPERATOR_DECISION_REGISTER.md`](../../qualification/sef/ASF_SEF_OPERATOR_DECISION_REGISTER.md) |
| ADR | [`../../../technical/adr/entries/2026-11-05/ADR-SEF-001.md`](../../../technical/adr/entries/2026-11-05/ADR-SEF-001.md) |
| Roadmap stage | [`../PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.7a **SEF-X** |
