> **Migrated (AUDIT-PROTOCOL-RESET-R2):** Historical plan-satellite audit register.
> **Original path:** docs\project\maintainers\plans\satellites\ELASTIC_CAPACITY_AND_SCALING_implementation_history.md
> **Original role:** Plan satellite — audit history + LC closeout
> **Canonical audit ownership:** docs/audit_results/ (this file is historical evidence only)

# ELASTIC_CAPACITY_AND_SCALING — audit history + LC closeout

**Parent hub:** [`ELASTIC_CAPACITY_AND_SCALING.md`](../ELASTIC_CAPACITY_AND_SCALING.md)

## Phase ECP-DOC — Domain pair establishment (Band 2an)

**Status:** **Done** (2026-06-08) — architecture + plan pair + ADR-SCALE-001; hub + audit routing updated  
**Prerequisites:** Phase W-OPS **Done** · Phase ORCH/FLOW backpressure **Done** · `kubernetes` integration beta  
**Goal:** Establish **19th domain pair** as canonical source for Harness Elastic Capacity Plane (ECP) — consolidate scattered scaling docs without runtime controller  
**Priority ladder:** **Band 2an** (§4.0 PLATFORM_FOUNDATION) — **closed** on doc merge  
**Execution order:** [§6.2an](.#62an-phase-ecp-doc-execution-order-band-2an--closed) · queue: [§6.1an](.#61an-harness-implementation-queue--elastic-capacity-domain-pair-closed)

**Delivery rule:** ECP-DOC.* = docs + ADR only; runtime work routes to ECP-DEPTH.*

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| ECP-DOC.1 | **`architecture/ELASTIC_CAPACITY_AND_SCALING.md`** — full ECP canon | **Done** | **Critical** | `docs/project/architecture` | Hub links; audit §30 extension |
| ECP-DOC.2 | **`plan/ELASTIC_CAPACITY_AND_SCALING.md`** — this file; ECP-DEPTH register | **Done** | **Critical** | `docs/project/maintainers/plans` | 1:1 pair check green |
| ECP-DOC.3 | **`docs/project/technical/adr/entries/2026-06-08/ADR-SCALE-001.md`** — ECP vs K8s HPA; tier separation | **Done** | High | `docs/project/technical/adr` | Linked from architecture + adr README |
| ECP-DOC.4 | **Hub update** — 19 domain pairs; audit routing for capacity | **Done** | High | `intergrax_runtime_architecture.md` | `check_docs_domain_pairs.py` OK |
| ECP-DOC.5 | **Cross-ref sync** — ORCHESTRATION §49, OBS §9.3, INTEGRATIONS k8s, AGENTS.md, audit map §30 | **Done** | High | `docs/*` | No orphan scaling narrative |
| ECP-DOC.6 | **Gate script** — `python scripts/docs/check_docs_domain_pairs.py` | **Done** | Medium | CI scripts | 19 pairs reported |

---

## Phase ECP-DEPTH — Elastic capacity runtime scaffold (Band 2ao — closed)

**Status:** **Done** (2026-06-09) — **28/28 scaffold Done** (ECP-6.2 **Cancelled**)  
**Honest outcome:** Contracts, `runtime/capacity`, gate tests, disabled-by-default host wiring — **not** production fleet autoscaling.
**Prerequisites:** Phase ECP-DOC **Done**  
**Goal (achieved):** Harness **L2** — typed control plane scaffold + CI evidence  
**Production elasticity:** Phase **ECP-PROD** (below)  
**Priority ladder:** **Band 2ao** — **closed** on scaffold; active queue = **ECP-PROD**  
**Traceability:** [Appendix A](.#appendix-a--elastic-capacity-traceability-phase-ecp-depth)

**Delivery rule:** One **ECP-* ID per PR** → update master table + architecture §22 → `pytest -m gate` green.

**Principle:** **complement, not replace** K8s HPA · async scheduler · policy-first scale-up · idempotent actions · hysteresis on scale-down.

**Out of scope:** K.1/K.2 product hosts · training/inference cluster autoscaling (MLOps) · replacing Tier-3 Helm ownership · agent registry mutation as “scaling”.

### ECP-DEPTH — Maturity targets

| Area | Current (post ECP-DOC) | Target | Primary IDs |
|------|------------------------|--------|-------------|
| ScalingProfile | L0 | L3 | ECP-1.* |
| Signal collection | L1 partial | L3 | ECP-2.* |
| Rules evaluator | L0 | L3 | ECP-3.* |
| K8s scale API | L1 beta | L3 | ECP-4.* |
| Celery worker scale | L2 manual | L3 | ECP-5.* |
| nginx / ingress | L0 | L2 | ECP-6.* |
| Policy + HITL gates | L0 | L3 | ECP-7.* |
| AHI ↔ ECP bridge | L0 | L2 optional | ECP-8.* |
| Capacity observability | L1 | L3 | ECP-OBS.* |

**Success gate (scaffold — met):** P0 + P1 contract tasks **Done**; unit/gate tests with mock K8s; `ScalingProfile` on environment profile.

**Not met (deferred to ECP-PROD):** live `GRAPH_BACKPRESSURE` → scale on hosts; production K8s/Celery adapters; FAUDIT §30 **production elasticity L3+**.

```text
Wave ECP0 — Package scaffold (4 tasks)
Wave ECP1 — Contracts + ScalingProfile P0 (5 tasks)
Wave ECP2 — Signal collector P0 (4 tasks)
Wave ECP3 — Evaluator + rules P0 (4 tasks)
Wave ECP4 — K8s provisioner P1 (4 tasks)
Wave ECP5 — Celery / queue worker scale P1 (3 tasks)
Wave ECP6 — nginx / ingress RFC + slug P2 (2 tasks)
Wave ECP7 — Policy, HITL, anti-flap P1 (3 tasks)
Wave ECP8 — AHI bridge optional P2 (1 task)
Wave ECP-OBS — Metrics + trace events P1 (2 tasks)
Total ECP-DEPTH: 28 (excluding ECP-DOC)
```

---

### 6.2an Phase ECP-DOC execution order (Band 2an — closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | ECP-DOC.3 | ADR-SCALE-001 | High |
| 2 | ECP-DOC.1 | Architecture canon | Critical |
| 3 | ECP-DOC.2 | Plan register | Critical |
| 4 | ECP-DOC.4 | Hub 19-pair index | High |
| 5 | ECP-DOC.5 | Cross-ref sync | High |
| 6 | ECP-DOC.6 | Domain pair gate | Medium |

### 6.1an Harness implementation queue — Elastic capacity domain pair (closed)

**Status:** **Closed** (2026-06-08)  
**Band:** 2an  
**Outcome:** 19th domain pair live; ECP canon + ADR accepted.

---

### 6.2ao Phase ECP-DEPTH execution order (Band 2ao — planned)

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| ECP0 | ECP-0.1–ECP-0.4 | 4 | Package scaffold + import gate |
| ECP1 | ECP-1.1–ECP-1.5 | 5 | **P0** — contracts + ScalingProfile |
| ECP2 | ECP-2.1–ECP-2.4 | 4 | **P0** — CapacitySignalCollector |
| ECP3 | ECP-3.1–ECP-3.4 | 4 | **P0** — ScalingEvaluator |
| ECP4 | ECP-4.1–ECP-4.4 | 4 | **P1** — K8s provisioner depth |
| ECP5 | ECP-5.1–ECP-5.3 | 3 | **P1** — Celery worker autoscale |
| ECP6 | ECP-6.1–ECP-6.2 | 2 | **P2** — nginx / ingress slug |
| ECP7 | ECP-7.1–ECP-7.3 | 3 | **P1** — Policy + HITL + flap guard |
| ECP8 | ECP-8.1 | 1 | **P2** — AHI approved proposal bridge |
| ECPOBS | ECP-OBS.1–ECP-OBS.2 | 2 | **P1** — Trace + metrics |
| **Total** | | **28** | |

---

## Phase ECP-PROD — Production elasticity (Band 2aú — closed)

**Status:** **Done** (2026-06-12) — ECP-PROD.1–7 **Done**; live cluster requires `INTERGRAX_KUBERNETES_URL` at runtime  
**Prerequisites:** ECP-DEPTH scaffold **Done**  
**Goal:** Close architecture §22 gap register — **L2 → L3+** production closed-loop capacity  
**Priority:** **P0** (ECP-PROD.1–3) · **P1** (ECP-PROD.4–7)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-PROD.1 | **Live signal bridge** — `GRAPH_BACKPRESSURE` bus → collector; optional `task_index` depth | **Done** | **Critical** | `capacity/event_bridge.py`, `scaling_wiring.py` | Unit gate |
| ECP-PROD.2 | **Scheduler governance** — skip apply on `hitl_required` / `denied` | **Done** | **Critical** | `capacity/scheduler.py` | Unit test HITL |
| ECP-PROD.3 | **K8s REST scale** — default factory scales Deployment via API | **Done** | **Critical** | `kubernetes/rest_client.py`, `p5/factories.py` | Gate test + mock HTTP |
| ECP-PROD.4 | **Celery worker scale** — provisioner calls adapter (not `pass`) | **Done** | High | `capacity/provisioner.py` | Gate test |
| ECP-PROD.5 | **Ceiling raise** — bounded `max_inflight_nodes` patch | **Done** | High | `capacity/ceiling_patcher.py` | Unit test |
| ECP-PROD.6 | **HITL approval path** — scale-up waits for operator | **Done** | High | `capacity/approval_queue.py`, `governance.py` | Queue + SCALE_* events |
| ECP-PROD.7 | **E2E gate** — backpressure → evaluate → K8s scale (mock) | **Done** | High | `tests/integration/runtime/test_ecp_backpressure_scale.py` | `-m gate` |
| AUDIT-IDEAL-30.4 | **Re-close** with real adapter contracts (not InMemory-only) | **Done** | P2 | `production_adapters.py` | Live K8s when URL set; in-memory in CI |

### ECP-PROD — Sprint plan

| Sprint | Scope | Done when |
|--------|-------|-----------|
| **S1** | ECP-PROD.1–2 — signal bridge + scheduler HITL | Tests green; architecture §22 rows updated |
| **S2** | ECP-PROD.3 — K8s REST scale factory | Mock HTTP gate; no live cluster in CI |
| **S3** | ECP-PROD.4–5 — Celery + ceiling provisioner | Provisioner no longer no-op for shipped kinds |
| **S4** | ECP-PROD.6–7 + AUDIT-IDEAL-30.4 | E2E integration; production adapter gate honest |

---

*End of Elastic Capacity and Scaling Implementation Plan.*

---

## Phase ECP-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates ECP-DOC + ECP-PROD + AUDIT-IDEAL-30.1/30.4; no open P0/P1  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| ECP-LC-S1 | **Re-audit** — ECP register + honest maturity verdict | **Done** | High | No P0/P1 |
| ECP-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| ECP-LC-S3 | **Gate verification** | **Done** | High | 17/18 capacity tests · `check_production_capacity_adapters` |
| ECP-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** `test_capacity_approval_queue_flow` event assertion flake · live K8s soak · nginx/ingress slug

**Audit note (2026-06-18):** capacity suite **18/18 green** in revalidation; flake row retained for CI stability hardening.

### 6.1av Harness implementation queue — Elastic capacity audit maintenance (planned)

**Source:** Layer 20 audit (2026-06-18) — `ELASTIC_CAPACITY_AND_SCALING` · [`../audit_results/2026-06-18/ELASTIC_CAPACITY_AND_SCALING.md`](../audit_results/2026-06-18/ELASTIC_CAPACITY_AND_SCALING.md)  
**Priority ladder:** **Band 1** (§6.1) — test stability + ops depth; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **ECP-MAINT-01** | Test | P2 | **Done** | Harden `test_capacity_approval_queue_flow` — deterministic `scale_requested` event assertion | Event-kind assertion; no timing flake |
| 2 | **ECP-MAINT-02** | Ops | P3 | **Done** | Live K8s soak gate — nightly or manual runbook | Manual runbook row in architecture |
| 3 | **ECP-MAINT-03** | Cross-ref | P4 | **Done** | nginx/ingress slug — cross-ref [`INT-MAINT-04`](INTEGRATIONS.md#61av-harness-implementation-queue--integrations-audit-maintenance-planned) | ECP architecture ingress bridge |
| 4 | **ECP-MAINT-04** | CI | P3 | **Done** | Register capacity suite in AGENTS.md verification list alongside `check_production_capacity_adapters` | AGENTS.md gate bundle updated |

**Suggested PR order:** ECP-MAINT-01 → ECP-MAINT-04 → ECP-MAINT-02 → ECP-MAINT-03.

---

*End of Elastic Capacity and Scaling Implementation Plan.*
