**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix G — Harness production audit traceability (Phase U)

**Source:** Harness-system audit (2026-06-01) — lab/Tier-1/Tier-3 only; **no business agents**.  
**Goal:** Map every finding to exactly one Phase U deliverable. Update **Status** when **Done** / **Won't fix** (with reason).  
**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### G.1 Security (P0)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| SEC-01 | Lab `POST /v1/lab/run` and `/debug/*` without authentication | U-Sec.1 | Done |
| SEC-02 | MCP enabled by default (`LAB_INCLUDE_MCP=true`) — second open surface | U-Sec.2 | Done |
| SEC-03 | `sandbox.exec` enabled in default lab tool profile | U-Sec.3 | Done |
| SEC-04 | `harness_production_mode()` always `False` — no strict production path | U-Sec.4 | Done |

### G.2 Contracts & policy (P1)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| CON-01 | `Agent` (ABC) vs `UAEPAgent` (Protocol) — no unified inheritance | U-Con.1 | Done |
| CON-02 | `RuntimePolicyBundle` built in lab ctx but not applied to `RuntimeConfig` | U-Pol.1 | Done |
| CON-03 | `PolicyEngine` (NexusLoop) vs `policy_bundle` (RuntimeConfig) — dual systems | U-Pol.2 | Done |
| CON-04 | `ToolPlanningService` imports `ToolsAgentConfig` from Tier-0 `tools_agent` | U-Typ.2 | Done |
| CON-05 | `runtime_state` uses `isinstance(CatalogToolPlanner)` not protocol | U-Typ.3 | Done |
| CON-06 | `create_lab_interaction_adapter()` uses `IntegrationProfile.lab()` not preset | U-Arch.1 | Done |
| CON-07 | Skill `skill_ids` resolved at register — no runtime E2E proof in gate | U-Con.3 | Done |

### G.3 Typing & hygiene (P2)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| TYP-01 | `ToolsAgentConfig` tuple bug (`temperature = None,`) | U-Typ.1 | Done |
| TYP-02 | `RuntimePolicyBundle.budget` / `plan_loop` typed as `Any` | U-Pol.3 | Done |
| TYP-03 | `# type: ignore` on lab integration wiring adapters | U-Arch.2 | Done |
| TYP-04 | `getattr` outside harness audit (tools_agent prune, profile, sandbox) | U-Typ.4 | Done |
| TYP-05 | `hasattr` on harness paths (shared_task_context, engine_plan, platform_wiring) | U-Typ.5 | Done |
| TYP-06 | `ToolPlanDecision` vs `AgentDecision` naming collision risk | U-Leg.3 | Done |

### G.4 Legacy & naming (P3)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| LEG-01 | `tools_agent_answer` and ToolsAgent naming in Tier-1 runtime | U-Arch.3 | Done |
| LEG-02 | `ToolsAgent.run` still full orchestrator — deprecation incomplete | U-Leg.1 | Done |
| LEG-03 | `rag.answers` module remains; tests filtered not removed | U-Leg.2 | Done |
| LEG-04 | Legacy tool plan booleans (`from_legacy`, `uses_legacy_rag_flag_only`) | U-Leg.3 | Done |

### G.5 Documentation & CI (P4)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| DOC-01 | `guides/HARNESS_ENVIRONMENT.md` claims policy bundle wired — lab does not apply bridge | U-Doc.1, U-Pol.1 | Done |
| DOC-02 | Phase K footer still "after Phase S" in harness docs | U-Doc.3 | Done |
| CI-01 | harness-smoke omits Phase T unit tests | U-CI.1 | Done |
| CI-02 | No acceptance test for strict production harness path | U-CI.2 | Done |
| CI-03 | harness-smoke vs gate run on different OS images | U-CI.3 | Done |

### G.6 Phase U paydown log

| Date | U ID | Summary |
|------|------|---------|
| 2026-06-01 | U.0.* | Appendix G + Phase U section added to implementation plan (audit → backlog) |
| 2026-06-02 | §4.1 | Harness completion: U-Leg.1–3, U-Arch.2, U-Typ.4, U-CI.3, harness.skill_registry, research UAEP parity; gate **481** |
| — | — | *(append row per merged PR)* |

**Coverage target:** Phase U + §4.1 harness completion backlog **Done** (2026-06-02). **K.1/K.2 deferred** until product prioritization.

---
