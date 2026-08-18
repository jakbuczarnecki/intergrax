# Reliability Failure And Hitl — Implementation Plan

**Architecture (1:1):** [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](../../architecture/RELIABILITY_FAILURE_AND_HITL.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

**Last updated:** 2026-06-20 — **P2-ARCH-09** Attempt Ledger canon.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (RELIABILITY_FAILURE_AND_HITL plan).

- **Implement / audit default:** §6.1 REL / HITL maintenance · open retry/HITL rows · closed reliability LC — satellite only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](../../architecture/RELIABILITY_FAILURE_AND_HITL.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Architecture documentation (P2)

| ID | Task | Status |
|----|------|--------|
| **P2-ARCH-09** | Clarify attempt ledger and retry ownership | **Done** (2026-06-20) |

Architecture: [`RELIABILITY_FAILURE_AND_HITL.md`](../../architecture/RELIABILITY_FAILURE_AND_HITL.md#attempt-ledger).

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.8 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-22.1 | §22 Reliability | Compensation flows on product side-effect paths | P1 | **Done** |
| AUDIT-IDEAL-22.2 | §22 Reliability | Partial results contract on all reference hosts | P2 | **Done** |
| AUDIT-IDEAL-6.5 | §6 LLM (shared) | Profile failover chain on retriable provider errors | P1 | **Done** — [M-LLM-X.4.1–4.4](plan/LLM_ADAPTERS.md) |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

### 6.2bl Phase REL execution order (Band 2u — closed 2026-06-02)

**Status:** **Done** · register: [Phase REL](RELIABILITY_FAILURE_AND_HITL.md) · queue: [§6.1o](.#61o-harness-implementation-queue--reliability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REL-DOC.1 | Appendix R + plan sync | High |
| 2 | REL-1 | `reliability_runtime_bridge` + `reliability_wiring` | Critical |
| 3 | REL-2 | `reliability_assembly_resolver` | High |
| 4 | REL-3 | `check_harness_reliability_wiring.py` | Medium |### 6.2bk Phase OBS execution order (Band 2t — closed 2026-06-02)

**Status:** **Done** · register: [Phase OBS](plan/OBSERVABILITY.md) · queue: [§6.1n](.#61n-harness-implementation-queue--observability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | OBS-DOC.1 | Appendix Q + plan sync | High |
| 2 | OBS-1 | `observability_runtime_bridge` + `observability_wiring` | Critical |
| 3 | OBS-2 | `observability_assembly_resolver` | High |
| 4 | OBS-3 | `check_harness_observability_wiring.py` | Medium |

---

## Phase REL — Reliability control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REL-DOC.1 + REL-1–3)

**Audit basis:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../../audit_results/AUDIT_PROTOCOL.md) §22; H-APP `ReliabilityProfile` **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix R**.

**Priority ladder:** **Band 2u** (§4.0) — closed; default queue = **§6.1** maintenance.

### REL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REL-DOC.1 | REL0 | **Appendix R** — reliability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REL-1 | REL1 | **`reliability_runtime_bridge`** + **`reliability_wiring`** | **Done** | `reliability_runtime_bridge.py`, `reliability_wiring.py`, `runtime_config_bridge.py` | `test_harness_reliability_wiring.py` |
| REL-2 | REL2 | **`reliability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `reliability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| REL-3 | REL3 | **Host reliability CI** — `check_harness_reliability_wiring.py` | **Done** | `scripts`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only retry/fallback policies — [§6.3a](.#63a-business-backlog-register-consolidated).

---

---

### Phase F — Advanced / On-Demand

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| F.1 | ShadowWorkspace | **Done** | `runtime/workspace`; UAEP + NexusLoop integration |
| F.2 | SandboxRuntime | **Done** | `runtime/sandbox`; `sandbox.exec` via BoundToolGateway |
| F.3 | Advanced HITL (reject/escalation store) | **Done** | `runtime/human` store + NexusLoop reject/escalate |
| F.4 | Long-running tasks / Slack-Teams | **Done (partial)** | Checkpoints ✅; Slack/Teams = notification stub only |

| F.5 | Typed task contract | **Done** | `TaskExecutionOptions`, `TaskRuntimeState`, `TaskResultSummary`, bridge |

Long-running **full** §26 (scheduler, UAEP mid-step) and Slack/Teams **full** §18 — see Phase G–H below.

---

## Phase REL-ADV — Resilience policies and autonomy slider (closed)

**Status:** **Done** (2026-06-09) — architecture canon §34–§35; runtime REL-ADV.1–6 implemented.

**Goal:** Unify distributed retry/recovery behaviour under a composable `ResiliencePolicy` model and expose user-facing **AutonomyLevel** (manual / ask / autonomous) with mid-run changes.

**Prerequisites:** Phase REL **Done**; UAEP §42.8–§42.10 **Done**; PolicyEngine wiring **Done**.

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REL-ADV-DOC.1 | REL-ADV0 | Canon sync — architecture §34–§35, UAEP §42.10.2 | **Done** | `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` | Hub index + cross-refs |
| REL-ADV.1 | REL-ADV1 | **`ResiliencePolicy`** Pydantic model + profile field on `ReliabilityProfile` | **Done** | `contracts/resilience_policy.py`, `environment_profile.py` | `test_policy_resolver.py` |
| REL-ADV.2 | REL-ADV2 | **Policy resolver** — map failure class → policy action (reboot strategies) | **Done** | `runtime/resilience/policy_resolver.py`, `retry_engine.py` | `test_policy_resolver.py` |
| REL-ADV.3 | REL-ADV3 | **`AutonomyLevel`** on `TaskExecutionOptions` + effective level middleware | **Done** | `autonomy_resolver.py`, `autonomy_middleware.py` | `test_autonomy_resolver.py` |
| REL-ADV.4 | REL-ADV4 | **Mid-run autonomy API** — set level on active task | **Done** | `harness_task_routes.py`, `task_control.py` | `ActiveTaskRegistry` + HTTP route |
| REL-ADV.5 | REL-ADV5 | **Trace events** — `AUTONOMY_LEVEL_*`, `RECOVERY_REBOOT` | **Done** | `runtime_event.py`, `phase_coverage.py` | `test_schema_registry_b07.py` |
| REL-ADV.6 | REL-ADV6 | **CI** — `check_harness_resilience_policy.py` | **Done** | `scripts` | lab host audit OK |
| REL-ADV.7 | Tier-3 | **Product host parity** — reliability enricher + autonomy HTTP on scaffold-opt-in hosts | **Done** | H-APP-WIRING.1 **Done** | `UnifiedTaskRunner(task_enricher=…)` |

**ADR policy:** REL-ADV.1 → ADR-REL-001 (resilience policy unification) when implementation starts; REL-ADV.3 → no ADR if enum-only on existing PolicyEngine path.

**Audit note (2026-06-09):** REL-ADV.1–6 **Done** at runtime; REL-ADV.7 tracks lab-only HTTP exposure per [`architecture/ORCHESTRATION.md`](../../architecture/ORCHESTRATION.md) §59.4.

**Explicitly excluded:** K.1/K.2 product policies; OS-level process supervisor (use ECP / host ops).

---

## Phase IDEAL-L3 — Reliability depth (Band 2ax)

**Register:** [`plan/IDEAL_HARNESS_L3.md`](IDEAL_HARNESS_L3.md) · queue: [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) §6.1at

| ID | Deliverable | Status |
|----|-------------|--------|
| IDEAL-22.1 | `harness_error_taxonomy.py` + expanded `ErrorClassifier` | **Done** |
| IDEAL-22.2 | Quality vs dependency recovery paths | **Done** |
| IDEAL-22.3–22.6 | Compensation, partial results, chaos, per-step retry | Planned (W2) |

**Gate:** `tests/unit/runtime/architecture/test_ideal_harness_l3_depth_gate.py`

---

## Phase RELIABILITY-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates REL + REL-ADV + AUDIT-IDEAL-22.1/22.2; no open P0/P1  
**Prerequisites:** Phase REL **Done** · REL-ADV **Done**  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| REL-LC-S1 | **Re-audit** — REL/REL-ADV register + HITL verdict | **Done** | High | No P0/P1 |
| REL-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| REL-LC-S3 | **Gate verification** | **Done** | High | 23 tests · 3 CI scripts |
| REL-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_legacy campaign README` mature |

**Deferred P2–P4:** IDEAL-22.3–22.6 chaos/per-step retry · ResiliencePolicy HTTP product parity · durable async queue opt-in · M-LLM-X.4 failover (LLM domain)

### 6.1av Harness implementation queue — Reliability audit maintenance (planned)

**Source:** Layer 17 audit (2026-06-18) — `RELIABILITY_FAILURE_AND_HITL` layer 22 · [`../audit_results/2026-06-18/RELIABILITY_FAILURE_AND_HITL.md`](../../../audit_results/2026-06-18/RELIABILITY_FAILURE_AND_HITL.md)
**Priority ladder:** **Band 1** (§6.1) — IDEAL-L3 W2 depth + cross-domain wiring; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **REL-MAINT-01** | Code/Test | P2 | **Done** | IDEAL-22.3–22.6 — compensation + partial results + chaos harness + per-step retry depth | `test_rel_maint_depth.py`; cross-ref FLOW-MAINT-01 |
| 2 | **REL-MAINT-02** | Code | P2 | **Done** | ResiliencePolicy HTTP surfaces on product hosts (beyond lab_stack) | `check_harness_resilience_policy.py` in verification bundle |
| 3 | **REL-MAINT-03** | Cross-ref | P2 | **Done** | Durable async queue opt-in — cross-ref [`ORCH-MAINT-04`](ORCHESTRATION.md#61av-harness-implementation-queue--orchestration-audit-maintenance-planned) | REL architecture documents HITL/retry interaction |
| 4 | **REL-MAINT-04** | Cross-ref | P2 | **Done** | M-LLM-X.4 profile failover — cross-ref [`LLM-MAINT-03`](LLM_ADAPTERS.md#61av-harness-implementation-queue--llm-adapters-audit-maintenance-planned) | Failover documented in REL architecture |

**Suggested PR order:** REL-MAINT-01 → REL-MAINT-02 → REL-MAINT-03 → REL-MAINT-04.

**Cross-domain:** FLOW-MAINT-01 · ORCH-MAINT-04 · LLM-MAINT-03.

---

*End of Reliability, Failure Model, and HITL Implementation Plan.*
