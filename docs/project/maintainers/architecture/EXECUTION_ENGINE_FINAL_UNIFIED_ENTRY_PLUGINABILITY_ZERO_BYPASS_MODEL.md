# Execution Engine — Final Unified Entry, Pluginability & Zero-Bypass Model (EE-FINAL-ARCH)

**Classification:** `MAINTAINER_CERTIFICATION`  
**Status:** `CERTIFIED` (audit EE-FINAL-ARCH on `development`)  
**Audience:** Maintainers, enterprise qualification, architecture gates  

**Semantic authority:** [`EXECUTION_ENGINE_OWNERSHIP_MODEL.md`](EXECUTION_ENGINE_OWNERSHIP_MODEL.md) (EE-A1), [`PLATFORM_EXECUTION_UNIFICATION_ARCHITECTURE.md`](PLATFORM_EXECUTION_UNIFICATION_ARCHITECTURE.md), [`UNIFIED_EXECUTION_ARCHITECTURE.md`](../../architecture/UNIFIED_EXECUTION_ARCHITECTURE.md).  

**Frozen entry inventory SSOT:** [`../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md`](../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md) (22 EP rows; BYPASS=0).  

**Qualification record:** [`../qualification/EE_FINAL_ARCH_UNIFIED_ENTRY_PLUGINABILITY_ZERO_BYPASS_CERTIFICATION.md`](../qualification/EE_FINAL_ARCH_UNIFIED_ENTRY_PLUGINABILITY_ZERO_BYPASS_CERTIFICATION.md)

---

## 1. Canonical execution flow

All **supported platform execution** converges on one authoritative boundary (multiple host-level entry **functions** are allowed; one **execution authority** is not):

```text
Applications · Scenarios · Agents · Decision · Workflows · Plugins · Integrations · HITL · Nexus · Tools
        │
        ▼
PLATFORM EXECUTION ENTRY (intent / admission — not lifecycle owner)
        │
        ▼
ExecutionRuntime                    ← sole root lifecycle owner
        │
        ▼
ExecutionBoundary                   ← identity propagation
        │
        ▼
StrategyExecutionRouter
        │
        ├── Nexus orchestration (GraphExecutor / NexusLoop)
        ├── ChildExecutionRunner    ← ChildExecutionPort
        └── RuntimeToolInvoker      ← governed tool / side-effect seam
```

**Convergence rule:** API, CLI, scenario, queue worker, and hosted application factories may each expose an entry surface, but production work that performs execution must reach `ExecutionRuntime` via `HostTaskExecutionPort` / `Execution` facade (NPSC-3C-D, P0 inventory).

---

## 2. Final owner matrix

| Concern | Canonical owner | Duplicate authoritative owner |
| ------- | ----------------- | ----------------------------- |
| Lifecycle | `ExecutionRuntime` | **0** (forbidden) |
| Identity mint | `ExecutionIdentityAuthority` / `identity_authority.py` | **0** |
| Governance evaluation | Decision + policy plane → authorization | **0** execution schedulers |
| Authority propagation | `ExecutionBoundary` + delegation contracts | **0** |
| Orchestration / scheduling | Nexus (`GraphExecutor`, `NexusLoop`) | **0** second runtime loops |
| Child execution | `ChildExecutionRunner` | **0** parent→specialist direct calls |
| Tools / external side effects | `RuntimeToolInvoker` | **0** agent-local raw executors |
| Recovery | NPSC-5E plane (`LongRunningCoordinator`, retry admission) | **0** recovery-owned execution |
| Evidence | NPSC-5F plane (`RuntimeEventPersistence`, export boundary) | **0** evidence-driven control |
| Diagnostics | Diagnostics plane (read / interpret) | **0** execute / retry / govern |
| Capacity / admission | EE-B1.2 ports (preview ≠ execution owner) | **0** |
| Shutdown | EE-B1.1 / EE-B4-B contracts | **0** |

---

## 3. Execution entry inventory

The **authoritative row-level inventory** remains the frozen P0 document (EP-01 … EP-22). EE-FINAL-ARCH does not fork rows; it certifies that:

| Metric | Value |
| ------ | ----: |
| Total execution-capable entrypoints | 22 |
| CANONICAL | 19 |
| LEGACY BUT NON-PRODUCTION | 3 |
| BYPASS | 0 |
| Supported execution bypasses (production) | 0 |

Representative canonical entries: interaction intake (EP-01), scenario baseline (EP-02), queue worker (EP-06/07), Nexus task (EP-09), graph orchestration (EP-10), child ports (EP-11–13), recovery re-entry (EP-19), HITL continuation (EP-18).

---

## 4. Bypass inventory

| Bypass class | Supported production count | Evidence |
| ------------ | -------------------------: | -------- |
| Execution bypass | 0 | P0 inventory, U5 gates, `test_ee_final_arch_zero_execution_bypass.py` |
| Governance bypass | 0 | NPSC-4.2, EE-B3-A |
| Identity bypass | 0 | EE-A2, `test_execution_identity_single_authority_gate.py` |
| Authority bypass | 0 | HARDENING-6, EE-B3-A |
| Scheduler bypass | 0 | Nexus governance gates |
| Tool bypass | 0 | U5 EP-14, `test_ee_final_arch_tool_side_effect_boundary.py` |
| Child execution bypass | 0 | U4 child closure |
| Recovery bypass | 0 | NPSC-5E final qualification |
| Persistence bypass | 0 | NPSC-5F + EEC-1 |
| Direct side-effect bypass | 0 | RuntimeToolInvoker seam |

Static production import scan (applications / agents) blocks direct imports of `ExecutionRuntime`, `NexusLoop`, `UnifiedTaskRunner`, and `provider_executor` outside explicit allowlisted composition seams.

---

## 5. Tool and side-effect inventory

| Path | Owner | Production? |
| ---- | ----- | ----------- |
| Declarative / catalog tools | `RuntimeToolInvoker` → `ToolExecutor` protocol → registry adapter | Yes (EP-14) |
| Compensation enqueue | Admitted tool path via invoker wiring | Yes |
| External operations (LLM, etc.) | `external_operations` admission + termination ports | Yes (under active execution) |
| Agent-local `RuntimeToolInvoker(` construction | — | **Forbidden** in `intergrax/agents` |

---

## 6. Scheduler ownership

Nexus owns **how / when / dependencies / fan-out** under an active execution context. `GraphExecutor` schedules graph steps and delegates children through `ChildExecutionRunner`. No second `ExecutionRuntime` is constructed inside `NexusLoop`. Long-running resume is owned by `LongRunningCoordinator` (recovery plane), not ad-hoc agent timers.

---

## 7. Persistence architecture

```text
core semantics → persistence port / contract → configured provider → vendor adapter
```

Execution core modules (`runtime.py`, `boundary.py`, `host_task.py`, …) do **not** import concrete event store implementations. Evidence and runtime events use `RuntimeEventPersistence` / `EvidencePersistencePort` boundaries (NPSC-5F). SQLite / in-memory stores are reference or test adapters, not authoritative architecture embedded in `ExecutionRuntime`.

---

## 8. Pluginability model

Plugins declare **capabilities and contracts** (manifest, admission, typed SPI). Registration alone does **not** grant execution authority. Core orchestration depends on **ports**, not concrete plugin modules (HARDENING-5, DS-PLUGIN gates). Forbidden pattern:

```text
plugin loaded → arbitrary direct runtime / provider access
```

Approved pattern:

```text
CORE → CONTRACT / PORT → ADAPTER / PLUGIN implementation
```

---

## 9. Provider abstraction model

| Domain | Port / contract | Adapter layer | Vendor in execution core? |
| ------ | --------------- | ------------- | ------------------------: |
| LLM | Inference profiles + external operation admission | `integrations/`, provider adapters | 0 |
| Search | Tool / integration contracts | integrations | 0 |
| Storage | Document / object ports | integrations | 0 |
| Persistence | Event / evidence persistence ports | `runtime/events/stores`, observability adapters | 0 |
| Tools | `ToolExecutor` protocol, registry | `runtime/nexus/tools`, agents persistence wiring | 0 |
| Observability | Export boundary, typed facts | Sink adapters | 0 |
| Identity | `ExecutionIdentityAuthority` | Composition only | 0 |
| Policy | Governance authorization contracts | Policy plugins | 0 |
| Capability | Registry + manifest | Plugin admission | 0 |

Vendor SDK imports in `intergrax/runtime/execution/*` core files: **0** (EE-FINAL-ARCH gate).

---

## 10. Accepted legacy and test-only paths

P0 **LEGACY BUT NON-PRODUCTION** rows (3): contract-only or test-wired loops (e.g. EP-17 work-stage capability loop) gated by U5 — no production `applications/` import of the loop module. UNSUPPORTED / DEAD: 0. Legacy allowlisting is **documented + gated**, not composition-wired.

---

## 11. Zero-bypass conclusion

On the certified `development` baseline:

```text
supported execution bypass = 0
governance bypass = 0
identity bypass = 0
authority bypass = 0
scheduler bypass = 0
tool bypass = 0
recovery bypass = 0
persistence bypass = 0
direct side-effect bypass = 0
vendor-specific authoritative execution core coupling = 0
```

**PRODUCTION CODE CHANGED for EE-FINAL-ARCH:** NO (documentation + architecture gates only).

Re-verify against GitHub `development` before downstream EE-FINAL enterprise sign-off.
