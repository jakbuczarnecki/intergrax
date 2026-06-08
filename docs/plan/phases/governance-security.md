# Implementation Phases — Governance Security

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Phase GOV-AUDIT — Governance control plane (audit closeout)

**Status:** **Done** (2026-06-05) — runtime governance via V-REM, H-APP, DX-5.8; documentation via GOV-DOC.*  
**Prerequisites:** Phase V-REM **Done**, H-APP.2.4–2.8 **Done**, DX-5.8 **Done**  
**Goal:** Close governance/policy/observability audit (AUDIT_MAP §5, §21) with a single authoring map and traceability — **no** new OS features.  
**Author map:** [`AGENT_CREATION_GUIDE.md` Appendix H](AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane)

**Delivery rule:** GOV-DOC.* = docs-only PRs; no code unless regression found → route to **REG-*** under §6.1.

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| GOV-DOC.1 | **Appendix H** — control plane map (profiles, bundles, hooks, EP groups, mandatory vs optional observability) | **Done** | High | `AGENT_CREATION_GUIDE.md` | TOC + §H.1–H.8 present |
| GOV-DOC.2 | **Cross-ref sync** — plan Documentation model, README, `HARNESS_ENVIRONMENT.md`, canon §42.11.5, AUDIT_MAP §5/§21, audit prompt ref #5 | **Done** | Medium | `docs/*` | Links resolve; no orphan audit layer |
| GOV-DOC.3 | **`EXTENSION_AUTHOR_GUIDE.md` §10** — `intergrax.policy_rules` author surface | **Done** | Medium | `EXTENSION_AUTHOR_GUIDE.md` | DX-5.8 traceability |
| GOV-PROD.1 | Unified product observability dashboard (beyond lab debug APIs) | **Deferred** | — | — | **§6.3** product decision; optional `observability_backend` remains harness path |

**Explicitly out of scope:** K.1/K.2 policy; product-specific legal/org policy fragments beyond lab reference YAML.

---

## Phase SEC — Security control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (SEC-DOC.1 + SEC-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §23; V-SEC / V-REM-SEC **Done**; author map: `AGENT_CREATION_GUIDE.md` **Appendix S**.

**Priority ladder:** **Band 2v** (§4.0) — closed; default queue = **§6.1** maintenance.

### SEC — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| SEC-DOC.1 | SEC0 | **Appendix S** — security control plane closeout | **Done** | `AGENT_CREATION_GUIDE.md` | TOC + verification table |
| SEC-1 | SEC1 | **`security_runtime_bridge`** + **`security_wiring`** | **Done** | `security_runtime_bridge.py`, `security_wiring.py`, `runtime_config_bridge.py` | `test_harness_security_wiring.py` |
| SEC-2 | SEC2 | **`security_assembly_resolver`** — profile ↔ middleware conformance | **Done** | `security_assembly_resolver.py`, `harness_host_runtime.py`, `nexus_factory.py` | assembly validation tests |
| SEC-3 | SEC3 | **Host security CI** — `check_harness_security_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only security dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---

## Phase COST — Cost governance control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (COST-DOC.1 + COST-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §24; V-COST **Done**; author map: `AGENT_CREATION_GUIDE.md` **Appendix T**.

**Priority ladder:** **Band 2w** (§4.0) — closed; default queue = **§6.1** maintenance.

### COST — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| COST-DOC.1 | COST0 | **Appendix T** — cost governance control plane closeout | **Done** | `AGENT_CREATION_GUIDE.md` | TOC + verification table |
| COST-1 | COST1 | **`CostProfile`** + **`cost_runtime_bridge`** + **`cost_wiring`** | **Done** | `environment_profile.py`, `cost_runtime_bridge.py`, `cost_wiring.py`, `policy_wiring.py` | `test_harness_cost_wiring.py` |
| COST-2 | COST2 | **`cost_assembly_resolver`** — profile ↔ budget conformance | **Done** | `cost_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py` | assembly validation tests |
| COST-3 | COST3 | **Host cost CI** — `check_harness_cost_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product FinOps dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---

