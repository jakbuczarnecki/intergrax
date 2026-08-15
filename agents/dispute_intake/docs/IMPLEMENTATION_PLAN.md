# dispute_intake agent — Implementation Plan

**The implementation map** for this Tier-2 agent — phases, status, gaps, and verification.

Status: Working draft (2026-06-07) — **Scaffold baseline**

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../docs/project/architecture/intergrax_runtime_architecture.md)
Agent workflow: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md)

Principle: **evolve, not rewrite** · **reuse Tier-0** · **no Tier-3 imports in agent code**

---

## Documentation model

Do not maintain separate status/readiness files under this agent. Use:

| Topic | Where |
|-------|--------|
| Purpose, contracts, I/O, runtime layout | **ARCHITECTURE.md** (this directory) |
| Task status, phases, next steps | **This file** |
| Significant agent architecture decisions | **`adr`** — [`adr/README.md`](adr/README.md) |
| Platform harness work | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` (gate maintenance) |
| UAEP / Nexus workflow | `docs/project/technical/guides/AGENT_CREATION_GUIDE.md` |

---

## 0. Scope at a glance

| Field | Value |
|-------|-------|
| Agent id | `dispute_intake` |
| Class | `DisputeIntakeAgent` |
| Primary capability | `dispute.intake` |
| Tier | Tier-2 (`agents/dispute_intake`) |
| Host wiring | Tier-3 application manifest (when mounted) |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| DISPUTE_INTAKE-1 | Replace scaffold stub in ``on_next_step` / cognitive pattern hooks` | Planned | High | One PR per domain step |
| DISPUTE_INTAKE-2 | Extend `prompts/system.md` for domain | Planned | Medium | Keep prompts versioned here |
| DISPUTE_INTAKE-3 | Register skills/tools on `contract.py` | Planned | Medium | See `docs/project/architecture/SKILLS.md` |
| DISPUTE_INTAKE-4 | Agent smoke test green | Done | High | `tests/test_dispute_intake_agent.py` |
| DISPUTE_INTAKE-5 | Mount in Tier-3 host (optional) | Planned | Medium | `AgentBinding.mount(DisputeIntakeAgent, ...)` |

---

## 2. Verification

```bash
uv run pytest agents/dispute_intake/tests -q
```

After host wiring:

```bash
uv run pytest applications/<app>_application/tests -q
```


---

## 3. Platform alignment

Business agents and product-only work remain **end of plan** unless explicitly reprioritized —
see platform [`§6.3`](../../../docs/project/maintainers/plans/PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).
