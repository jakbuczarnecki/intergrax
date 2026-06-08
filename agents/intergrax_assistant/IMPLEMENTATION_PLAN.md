# intergrax_assistant agent — Implementation Plan

**The implementation map** for this Tier-2 agent — phases, status, gaps, and verification.

Status: Working draft (2026-06-08) — **Scaffold baseline**

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform plan: [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md)  
Agent workflow: [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md)

Principle: **evolve, not rewrite** · **reuse Tier-0** · **no Tier-3 imports in agent code**

---

## Documentation model

Do not maintain separate status/readiness files under this agent. Use:

| Topic | Where |
|-------|--------|
| Purpose, contracts, I/O, runtime layout | **ARCHITECTURE.md** (this directory) |
| Task status, phases, next steps | **This file** |
| Significant agent architecture decisions | **`adr/`** — [`adr/README.md`](adr/README.md) |
| Platform harness work | `docs/INTERGRAX_IMPLEMENTATION_PLAN.md` §6.1 |
| UAEP / Nexus workflow | `docs/guides/AGENT_CREATION_GUIDE.md` |

---

## 0. Scope at a glance

| Field | Value |
|-------|-------|
| Agent id | `intergrax_assistant` |
| Class | `IntergraxAssistantAgent` |
| Primary capability | `platform.assist` |
| Tier | Tier-2 (`agents/intergrax_assistant/`) |
| Host wiring | Tier-3 application manifest (when mounted) |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| INTERGRAX_ASSISTANT-1 | Replace scaffold stub in `steps/pipeline.py` | Planned | High | One PR per domain step |
| INTERGRAX_ASSISTANT-2 | Extend `prompts/system.md` for domain | Planned | Medium | Keep prompts versioned here |
| INTERGRAX_ASSISTANT-3 | Register skills/tools on `contract.py` | Planned | Medium | See `docs/architecture/SKILLS.md` |
| INTERGRAX_ASSISTANT-4 | Agent smoke test green | Done | High | `tests/test_intergrax_assistant_agent.py` |
| INTERGRAX_ASSISTANT-5 | Mount in Tier-3 host (optional) | Planned | Medium | `AgentBinding.mount(IntergraxAssistantAgent, ...)` |

---

## 2. Verification

```bash
uv run pytest agents/intergrax_assistant/tests -q
```

After host wiring:

```bash
uv run pytest applications/<app>_application/<app>_application_tests -q
```


---

## 3. Platform alignment

Business agents and product-only work remain **end of plan** unless explicitly reprioritized —
see platform [`§6.3`](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63-end-of-plan--deferred-product-work-only).
