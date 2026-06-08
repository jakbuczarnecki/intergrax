# Reliability Failure And Hitl — Implementation Plan

**Architecture (1:1):** [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](../architecture/RELIABILITY_FAILURE_AND_HITL.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

### 6.2bl Phase REL execution order (Band 2u — closed 2026-06-02)

**Status:** **Done** · register: [Phase REL](plan/OBSERVABILITY.md) · queue: [§6.1o](#61o-harness-implementation-queue--reliability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REL-DOC.1 | Appendix R + plan sync | High |
| 2 | REL-1 | `reliability_runtime_bridge` + `reliability_wiring` | Critical |
| 3 | REL-2 | `reliability_assembly_resolver` | High |
| 4 | REL-3 | `check_harness_reliability_wiring.py` | Medium |### 6.2bk Phase OBS execution order (Band 2t — closed 2026-06-02)

**Status:** **Done** · register: [Phase OBS](plan/OBSERVABILITY.md) · queue: [§6.1n](#61n-harness-implementation-queue--observability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | OBS-DOC.1 | Appendix Q + plan sync | High |
| 2 | OBS-1 | `observability_runtime_bridge` + `observability_wiring` | Critical |
| 3 | OBS-2 | `observability_assembly_resolver` | High |
| 4 | OBS-3 | `check_harness_observability_wiring.py` | Medium |

---

## Phase REL — Reliability control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REL-DOC.1 + REL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §22; H-APP `ReliabilityProfile` **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix R**.

**Priority ladder:** **Band 2u** (§4.0) — closed; default queue = **§6.1** maintenance.

### REL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REL-DOC.1 | REL0 | **Appendix R** — reliability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REL-1 | REL1 | **`reliability_runtime_bridge`** + **`reliability_wiring`** | **Done** | `reliability_runtime_bridge.py`, `reliability_wiring.py`, `runtime_config_bridge.py` | `test_harness_reliability_wiring.py` |
| REL-2 | REL2 | **`reliability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `reliability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| REL-3 | REL3 | **Host reliability CI** — `check_harness_reliability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only retry/fallback policies — [§6.3a](#63a-business-backlog-register-consolidated).

---

---

### Phase F — Advanced / On-Demand

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| F.1 | ShadowWorkspace | **Done** | `runtime/workspace/`; UAEP + NexusLoop integration |
| F.2 | SandboxRuntime | **Done** | `runtime/sandbox/`; `sandbox.exec` via BoundToolGateway |
| F.3 | Advanced HITL (reject/escalation store) | **Done** | `runtime/human/` store + NexusLoop reject/escalate |
| F.4 | Long-running tasks / Slack-Teams | **Done (partial)** | Checkpoints ✅; Slack/Teams = notification stub only |

| F.5 | Typed task contract | **Done** | `TaskExecutionOptions`, `TaskRuntimeState`, `TaskResultSummary`, bridge |

Long-running **full** §26 (scheduler, UAEP mid-step) and Slack/Teams **full** §18 — see Phase G–H below.



---
