# Implementation Phases — Registry Capability

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Phase PE — Prompt registry control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (PE-DOC.* + PE-1–3); gate **623 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §17; V-REM-PE.1/PE.2 governance schema (**Done**); author map: `AGENT_CREATION_GUIDE.md` **Appendix M**.

**Priority ladder:** **Band 2p** (§4.0) — closed; default queue = **§6.1** maintenance.

### PE — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| PE-1 | PE1 | **`PromptProfile`** + `prompt_runtime_bridge` — `catalog_path` → `RuntimeConfig.prompt_catalog_path` | **Done** | `environment_profile.py`, `prompt_runtime_bridge.py`, `config.py` | `test_prompt_runtime_bridge.py` |
| PE-2 | PE2 | **`prompt_wiring`** — `resolve_prompt_registry()`, `PromptRegistryProtocol` | **Done** | `prompt_wiring.py`, `prompt_registry_protocol.py` | `test_prompt_wiring.py` |
| PE-3 | PE3 | **Environment wire** — `materialize_runtime_config`, `build_runtime_context_from_environment`, `ApplicationBuildContext.prompt_registry` | **Done** | `runtime_config_bridge.py`, `environment_wiring.py`, `runtime_context.py` | wiring tests + gate |
| PE-4 | PE4 | **Nexus injection** — `prompt_registry_resolver`; `tools_step`, `tool_planning_prompts`, `engine_plan_models`, `engine_planner_messages` use `RuntimeContext.prompt_registry` | **Done** | `prompt_registry_resolver.py`, Nexus steps/planner | `test_tools_step_prompt_registry.py` |
| PE-DOC.1 | PE0 | **Appendix M** — prompt registry control plane (§M.1–M.6) | **Done** | `AGENT_CREATION_GUIDE.md` | TOC + verification table |

**Residual:** none on Tier-3 host build path. Legacy YAML prompt assets (`chat_router*`, `tools_agent_*`) remain as catalog files only.

---

## Phase REG — Registry architecture control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REG-DOC.1 + REG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §19; capability graph V-CG **Done**; author map: `AGENT_CREATION_GUIDE.md` **Appendix O**.

**Priority ladder:** **Band 2r** (§4.0) — closed; default queue = **§6.1** maintenance.

### REG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REG-DOC.1 | REG0 | **Appendix O** — registry architecture control plane | **Done** | `AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REG-1 | REG1 | **`HarnessRegistrySnapshot`** + `registry_wiring` + `RegistrySnapshotProtocol` | **Done** | `registry_snapshot.py`, `registry_wiring.py` | `test_registry_wiring.py` |
| REG-2 | REG2 | **`registry_assembly_resolver`** — profile ↔ registry conformance at wire time | **Done** | `registry_assembly_resolver.py`, `environment_wiring.py` | `test_registry_wiring.py` |
| REG-3 | REG3 | **Host registry resolution CI** — `check_harness_registry_resolution.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), marketplace UI, Band 3 product hosts — [§6.3a](#63a-business-backlog-register-consolidated).

---

## Phase CG — Capability graph control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CG-DOC.1 + CG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §20; Phase V-CG **Done**; author map: `AGENT_CREATION_GUIDE.md` **Appendix P**.

**Priority ladder:** **Band 2s** (§4.0) — closed; default queue = **§6.1** maintenance.

### CG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CG-DOC.1 | CG0 | **Appendix P** — capability graph control plane | **Done** | `AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CG-1 | CG1 | **`capability_graph_wiring`** — environment subgraph from catalog + registry snapshot | **Done** | `capability_graph_wiring.py`, `capability_graph_protocol.py` | `test_capability_graph_wiring.py` |
| CG-2 | CG2 | **`capability_graph_assembly_resolver`** — wire-time catalog node validation | **Done** | `capability_graph_assembly_resolver.py`, `environment_wiring.py` | `test_capability_graph_wiring.py` |
| CG-3 | CG3 | **Host capability graph CI** — `check_harness_capability_graph_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only graph nodes — [§6.3a](#63a-business-backlog-register-consolidated).

---

