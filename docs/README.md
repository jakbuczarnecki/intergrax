# Intergrax — Documentation

**Last updated:** 2026-05-27

When documents conflict, **`intergrax_runtime_architecture.md`** is authoritative for architecture.
For **agent creation workflow**, **`AGENT_CREATION_GUIDE.md`** is the single source of truth.

---

## Start here

| If you want to… | Read |
|-----------------|------|
| **Create a new agent (full process)** | [**AGENT_CREATION_GUIDE.md**](AGENT_CREATION_GUIDE.md) ← **canonical** |
| Understand architecture | [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) |
| See implementation status / phases | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) |
| Check Agent OS readiness | [INTERGRAX_AGENT_OS_READINESS_PLAN.md](INTERGRAX_AGENT_OS_READINESS_PLAN.md) |
| Go/no-go before business agents | [RUNTIME_READY_FOR_BUSINESS_AGENTS.md](RUNTIME_READY_FOR_BUSINESS_AGENTS.md) |

---

## Operational documents (`docs/`)

| File | Role |
|------|------|
| [**AGENT_CREATION_GUIDE.md**](AGENT_CREATION_GUIDE.md) | **Single canonical guide** — scaffold, register, run, inspect, evaluate |
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) | Architecture canon (four tiers, UAEP, Nexus) |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) | Implementation plan — phases and deliverables |
| [INTERGRAX_AGENT_OS_READINESS_PLAN.md](INTERGRAX_AGENT_OS_READINESS_PLAN.md) | Phase L readiness assessment |
| [RUNTIME_READY_FOR_BUSINESS_AGENTS.md](RUNTIME_READY_FOR_BUSINESS_AGENTS.md) | Checklist before Problem Radar / Vendor Discovery |
| [INTERGRAX_IMPLEMENTATION_GAP_ANALYSIS.md](INTERGRAX_IMPLEMENTATION_GAP_ANALYSIS.md) | Baseline audit + live status |
| [experiment_guide.md](experiment_guide.md) | Redirect → AGENT_CREATION_GUIDE.md |

---

## Product ideas (not platform canon)

| Location | Role |
|----------|------|
| [key_components/](key_components/) | Planned product modules |
| [agents/legal/](../agents/legal/) | Legal agent roadmap |

## Archive (do not update)

| File | Superseded by |
|------|---------------|
| [archive/ARCHITECTURE.md](archive/ARCHITECTURE.md) | Canon §5.1 |
| [archive/agent_factory.md](archive/agent_factory.md) | Canon §42 (UAEP) |
| [archive/nexus.md](archive/nexus.md) | Canon §9 (NexusLoop) |

---

## Tier model (summary)

| Tier | Folder | Role |
|------|--------|------|
| Tier-0 | `intergrax/` | Platform |
| Tier-1 | `intergrax/runtime/` | Nexus Agent OS |
| Tier-2 | `agents/` | Agents |
| Tier-3 | `applications/` | Applications |

**Execution:** `Task` → `NexusLoop` → `AgentEngine` (UAEP).

---

## Rules

1. Architecture changes → update canon first, then the implementation plan.
2. **Workflow changes → update `AGENT_CREATION_GUIDE.md` only** (not experiment_guide or scattered READMEs).
3. Product roadmaps do not override the canon.
