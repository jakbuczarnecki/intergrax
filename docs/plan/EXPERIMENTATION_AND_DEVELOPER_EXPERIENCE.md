## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_embedded_detail.md`](plan/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


# Experimentation And Developer Experience — Implementation Plan

**Architecture (1:1):** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

**Last updated:** 2026-06-20 — **P2-ARCH-13** Experimentation/DX architecture vs implementation rules boundary.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE plan).

- **Implement / audit default:** Hub §6 · [`plan/plan/`](plan/plan/) satellites on demand. **On demand (one max):** [`plan/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_embedded_detail.md`](plan/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_embedded_detail.md). §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md).
- **Satellites:** at most **one** `plan/plan/` file per session unless RESUME cites more.

---

## Architecture doc alignment (P2-ARCH)

| ID | Scope | Status |
|----|-------|--------|
| **P2-ARCH-13** | Clarify Experimentation/DX architecture vs implementation rules boundary | **Done** (2026-06-20) |

---

## Phase DX-IDEA — Idea intake audit (Mode I)

**Source:** Operator workflow for auditing a **single harness or product idea** before implementation.  
**Bootstrap:** [`bootstrap/idea_audit.txt`](../bootstrap/idea_audit.txt) · **Orchestrator:** [`audit/IDEA_AUDIT_ORCHESTRATOR.md`](../audit/IDEA_AUDIT_ORCHESTRATOR.md) · **Cursor rule:** `.cursor/rules/intergrax-idea-audit.mdc`  
**Status:** **Done** — live chat audit; idea in operator message; durable record = architecture + plan update after operator approval (no sidecar files).

| ID | Gap | Priority | Status |
|----|-----|----------|--------|
| DX-IDEA-01 | Mode I indexed in hub, audit map, bootstrap README; architecture §43.2 surface | P2 | **Done** |
| DX-IDEA-02 | `scripts/check_idea_audit_bootstrap.py` — bootstrap ↔ orchestrator consistency gate | P3 | **Done** |
| DX-IDEA-03 | Natural-language idea intake via Cursor rule; bootstrap without USER CONFIG placeholders | P2 | **Done** |

**Delivery rule:** One **DX-IDEA-\*** ID per PR → update this table → `check_idea_audit_bootstrap.py` green.

**no ADR needed** — documentation and DX workflow only; no runtime contract change.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_appendices.md`](plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_appendices.md) | appendices |
| [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_audit_history.md`](plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## 5. Definition of Done (Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** — for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** — hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts



---

---

## Phase MVP-EVOL — MVP-to-product evolution layer (Band 2at — planned)

**Status:** **Done** (2026-06-09) — architecture canon §44; MVP-EVOL.1–6 implemented.

**Goal:** Deliver systematic **prototype → MVP → production** tooling: simulation harness, replay UX, KPI/satisfaction hooks, and promotion gate automation — competitive DX for product teams on Intergrax.

**Prerequisites:** Phase DX **Done**; Phase EVAL **Done**; lab host **Done**.

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| MVP-EVOL-DOC.1 | MVP0 | Canon §44 + hub cross-ref | **Done** | `docs/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | ORCHE §58 index |
| MVP-EVOL.1 | MVP1 | **Promotion gate script** — G0–G2 CI checks (runnable, eval baseline, policy) | **Done** | `scripts/check_mvp_promotion_gates.py` | G0–G2 OK |
| MVP-EVOL.2 | MVP2 | **Agent simulator CLI** — multi-agent failure/contention scenarios | **Done** | `intergrax/cli/mvp_evolution.py` | `intergrax mvp simulate` |
| MVP-EVOL.3 | MVP3 | **Trace replay** — reconstruct from trace store | **Done** | `intergrax/cli/mvp_evolution.py` | `intergrax mvp replay` |
| MVP-EVOL.4 | MVP4 | **Product KPI registry** — tenant-scoped metric definitions + export | **Done** | `product_kpi_registry.py` | unit tests deferred |
| MVP-EVOL.5 | MVP5 | **User satisfaction adapter** — thumbs / CSAT event schema + online eval bridge | **Done** | `user_satisfaction.py` | `test_user_satisfaction.py` |
| MVP-EVOL.6 | MVP6 | **Author guide appendix** — MVP evolution playbook | **Done** | `guides/AGENT_CREATION_GUIDE.md` Appendix X | TOC + scripts table |
| MVP-EVOL.7 | Exposure | **Tier-3 router optional** — HTTP endpoints for simulate/replay/KPI export (or document CLI-only canon) | **Done** | `mvp_evolution_routes.py` · lab `/v1/mvp/*` when `LAB_HARNESS=true` | CLI remains canonical |

**Cross-plan:** MVP-EVOL.2 ↔ ORCH CFG matrix; MVP-EVOL.5 ↔ OBS + EVAL online registry; promotion G4–G5 ↔ Phase V / W-OPS.

**Audit note (2026-06-09):** MVP-EVOL.1–6 **Done**; remaining debt is **exposure** (CLI vs product HTTP) — see [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §59.4.

**Explicitly excluded:** Product analytics SaaS UI; K.1/K.2 feature work.

---
