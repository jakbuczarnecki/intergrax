# Intergrax — Documentation

**Last updated:** 2026-05-27

When documents conflict, **`intergrax_runtime_architecture.md`** is authoritative.

---

## Operational documents (`docs/`)

| File | Role |
|------|------|
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) | **Canon** — architecture, four tiers (§5.1), UAEP (§42) |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) | Implementation plan — what to build next |
| [INTERGRAX_AGENT_OS_READINESS_PLAN.md](INTERGRAX_AGENT_OS_READINESS_PLAN.md) | Phase L — Agent OS readiness |
| [RUNTIME_READY_FOR_BUSINESS_AGENTS.md](RUNTIME_READY_FOR_BUSINESS_AGENTS.md) | Go/no-go checklist before business agents |
| [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) | Canonical workflow for new agents |
| [INTERGRAX_IMPLEMENTATION_GAP_ANALYSIS.md](INTERGRAX_IMPLEMENTATION_GAP_ANALYSIS.md) | Baseline audit (§1–13) + live status (§14–16) |
| [experiment_guide.md](experiment_guide.md) | How to run experiments (NexusLoop, debug API) |

## Product ideas (not platform canon)

| Location | Role |
|----------|------|
| [key_components/](key_components/) | Planned product modules — may be outdated relative to Nexus |
| [agents/legal/](../agents/legal/) | Legal agent roadmap and implementation plan |

## Archive (do not update)

| File | Superseded by |
|------|---------------|
| [archive/ARCHITECTURE.md](archive/ARCHITECTURE.md) | Canon §5.1, §7 (three-layer → four-tier model) |
| [archive/agent_factory.md](archive/agent_factory.md) | Canon §42 (AgentEngine / UAEP) |
| [archive/nexus.md](archive/nexus.md) | Canon §7.2, §9 (NexusLoop, Task) |
| [archive/ROADMAP-2026-02-engineering.md](archive/ROADMAP-2026-02-engineering.md) | DevOps infra backlog — separate from runtime plan |

---

## Tier model (summary)

| Legacy concept | Tier today | Folder |
|----------------|------------|--------|
| Layer 1 — Core Capabilities | Tier-0 Platform | `intergrax/` |
| Layer 2 — Nexus | Tier-1 Nexus Runtime | `intergrax/runtime/` |
| Layer 3 — agent | Tier-2 Agent | `agents/` |
| *(missing in old docs)* | Tier-3 Application | `applications/` |

**Execution:** `Task` → `NexusLoop` → `AgentEngine` (UAEP or legacy pipeline).

---

## Rules

1. Architecture changes → update the canon first, then the plan and gap analysis.
2. Do not duplicate the tier model outside canon §5.1.
3. Product roadmaps (`agents/*/ROADMAP.md`) do not override the canon.
