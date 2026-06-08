# Intergrax — Instructions for AI Coding Agents

> **Audience:** Cursor, Claude Code, Codex, Gemini, and other LLM agents working in this repository.
> **Canonical docs:** `docs/` — always prefer linking over duplicating.

---

## Project summary

**Intergrax** is an **Agent OS and Harness AI runtime** for building, orchestrating, and validating specialized AI agents in Python. It is a four-tier platform:

```text
Tier-0  intergrax/           Platform (integrations, tools, skills, LLM, RAG, memory)
Tier-1  intergrax/runtime/   Nexus — Agent Operating System
Tier-2  agents/              Specialized agent capabilities
Tier-3  applications/        Deployable product environments
```

**Strategic goal:** production-grade Harness AI aligned with modern Agent Engineering practice.  
**Source:** [docs/INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/INTERGRAX_DEVELOPMENT_STRATEGY.md)

**Documentation boundary:** `docs/intergrax_runtime_architecture.md` (hub) + `docs/architecture/` and `docs/INTERGRAX_IMPLEMENTATION_PLAN.md` + `docs/plan/` cover the **Harness / Agent OS platform** only. Each **business environment** (`applications/<product>/`) and **business agent** (`agents/<name>/`) has its own architecture and implementation plan — do not treat platform canon as the product deployment plan.

---

## Before you write code

1. Read [README.md — Start here](README.md#start-here) for documentation navigation
2. Read the architecture hub [docs/intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md), then the relevant domain doc in [docs/architecture/](docs/architecture/README.md)
3. Check phase status in [docs/INTERGRAX_IMPLEMENTATION_PLAN.md](docs/INTERGRAX_IMPLEMENTATION_PLAN.md) and [docs/plan/phases/](docs/plan/phases/)
4. Follow the work cycle in [docs/INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/INTERGRAX_DEVELOPMENT_STRATEGY.md):

```text
ANALIZA → OCENA ARCHITEKTURY → OCENA PLANU → PROPOZYCJA USPRAWNIEŃ
  → AKTUALIZACJA DOKUMENTACJI → IMPLEMENTACJA → WERYFIKACJA → WNIOSKI
```

---

## Hard rules (never violate)

### Tier dependency boundaries

```text
intergrax/       MUST NOT import from agents/ or applications/
agents/          MUST NOT import from applications/
applications/    MAY import from agents/ and intergrax/
```

### Agent creation

- **Never modify `intergrax/runtime/`** when creating Tier-2 agents
- Agents consume Tier-0 only through Nexus policy and `ToolRuntime` — no direct vendor SDK imports
- Canonical workflow: [docs/AGENT_CREATION_GUIDE.md](docs/AGENT_CREATION_GUIDE.md)
- Success metric: idea → first Nexus run in **under one hour**

### Documentation

- **One source of truth per topic** — update existing docs in `docs/`, do not create parallel guides
- Strategy → `docs/INTERGRAX_DEVELOPMENT_STRATEGY.md`
- Architecture hub → `docs/intergrax_runtime_architecture.md`
- Architecture domains → `docs/architecture/<domain>.md` (index: `docs/architecture/README.md`)
- Status/phases → `docs/INTERGRAX_IMPLEMENTATION_PLAN.md` + `docs/plan/phases/`
- Agent workflow → `docs/AGENT_CREATION_GUIDE.md`
- Harness AI terms → `docs/architecture/PLATFORM_FOUNDATION.md` §5.3 only
- Nexus execution flow (narrative + diagrams) → `docs/NEXUS_EXECUTION_FLOW_REFERENCE.md` · delegation ADR → `docs/adr/ADR-FLOW-001.md`

### Harness platform

- Default queue is **§6.1 maintenance only** — harness platform is complete
- Business agents (Phase K) are **end of plan** — do not start without explicit product decision
- Tier-1/2/3 work is **composition and wiring** of existing Tier-0 modules — no parallel universal mechanisms

---

## Task routing — what to read

| Task | Read first |
|------|------------|
| Create a new agent | [docs/AGENT_CREATION_GUIDE.md](docs/AGENT_CREATION_GUIDE.md) |
| Wire integrations | [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) |
| Add or use tools | [docs/TOOLS.md](docs/TOOLS.md) · `intergrax/tools/USAGE.md` |
| Add or use skills | [docs/SKILLS.md](docs/SKILLS.md) |
| Configure LLM providers | [docs/LLM_ADAPTERS.md](docs/LLM_ADAPTERS.md) |
| RAG / retrieval | [docs/architecture/RAG_AND_RETRIEVAL.md](docs/architecture/RAG_AND_RETRIEVAL.md) · [docs/AGENT_CREATION_GUIDE.md Appendix K](docs/AGENT_CREATION_GUIDE.md) |
| Memory / context / LTM | [docs/MEMORY_ARCHITECTURE.md](docs/MEMORY_ARCHITECTURE.md) · [docs/architecture/CONTEXT_ENGINEERING.md](docs/architecture/CONTEXT_ENGINEERING.md) · [Appendix G](docs/AGENT_CREATION_GUIDE.md#appendix-g--memory--rag-naming-phase-q) |
| New application (Tier-3) | `applications/USAGE.md` · `poc_template_application/` |
| Plugin / extension | [docs/EXTENSION_AUTHOR_GUIDE.md](docs/EXTENSION_AUTHOR_GUIDE.md) |
| Governance / policy / HITL | [docs/AGENT_CREATION_GUIDE.md Appendix H](docs/AGENT_CREATION_GUIDE.md) |
| Multi-agent graphs / Nexus execution flow | [docs/NEXUS_EXECUTION_FLOW_REFERENCE.md](docs/NEXUS_EXECUTION_FLOW_REFERENCE.md) · [docs/AGENT_CREATION_GUIDE.md Appendix I](docs/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) |
| Available agents (roster) | [agents/README.md](agents/README.md) |
| Available application environments | [applications/README.md](applications/README.md) |
| Harness audit | [docs/INTEGRAX_HARNESS_AUDIT_MAP.md](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) |
| L4 adaptive harness | [docs/ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md](docs/ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) |
| Critic / verification / LLM-as-judge | [docs/CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md](docs/CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md) · Phase CRIT-V · [plan/phases/evaluation-adaptive-critic.md](docs/plan/phases/evaluation-adaptive-critic.md) |
| Observability spine / bus / extension | [docs/OBSERVABILITY_ARCHITECTURE.md](docs/OBSERVABILITY_ARCHITECTURE.md) · [ADR-OBS-001](docs/adr/ADR-OBS-001.md) · [plan/phases/observability-reliability.md](docs/plan/phases/observability-reliability.md) |
| UAEP / execution runtime | [docs/architecture/UNIFIED_EXECUTION_RUNTIME.md](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) |
| Orchestration / graphs (canon) | [docs/architecture/ORCHESTRATION.md](docs/architecture/ORCHESTRATION.md) |

---

## Scaffold commands

```bash
python -m intergrax.scaffold new-agent <name> --capability domain.action
python -m intergrax.scaffold new-skill <skill_id>
python -m intergrax.scaffold new-stack <name>    # agent + application bundle
uv run intergrax doctor
```

---

## Verification (required after harness changes)

```bash
uv run pytest -m gate -q
python scripts/check_harness_no_getattr.py
uv run python scripts/check_observability_gates.py
```

For agent-only work:

```bash
uv run pytest agents/<agent>/tests/ -q
```

Full local suite: `scripts\test.bat unit` (Windows) or equivalent `uv run pytest`.

---

## Code style & conventions

- Match surrounding code — naming, types, imports, documentation level
- Minimize scope — smallest correct diff; no unrelated changes
- Python 3.12, managed by `uv` — see `pyproject.toml`
- Copyright header on new files: `© Artur Czarnecki. All rights reserved.`
- Comments only for non-obvious business logic
- Do not add tests unless requested or they cover real behavior

---

## Anti-patterns (do not do)

- Duplicating architecture canon in README, comments, or new markdown files
- Importing `agents/` or `applications/` from `intergrax/`
- Direct vendor SDK usage in Tier-2 agents
- Modifying Nexus runtime for agent-specific needs
- Creating new universal mechanisms when Tier-0 already provides one
- Starting Phase K business agents without explicit product prioritization
- Committing secrets (`.env`, credentials, API keys)

---

## Key paths

| Path | Contents |
|------|----------|
| `intergrax/runtime/nexus/` | Nexus Agent OS core |
| `intergrax/runtime/nexus/orchestration/` | Intake, planning, graph, HITL runners |
| `intergrax/integrations/` | Integration Library |
| `intergrax/tools/` | Tool Library |
| `intergrax/skills/` | Skill Library |
| `intergrax/llm_adapters/` | LLM provider adapters |
| `intergrax/rag/` | RAG engine |
| `intergrax/scaffold/` | Scaffolding CLI |
| `agents/` | Tier-2 agents — roster: [agents/README.md](agents/README.md) |
| `applications/` | Tier-3 application hosts — index: [applications/README.md](applications/README.md) (LKW, DSW, legal, research, lab) |
| `docs/` | Canonical documentation |
| `tests/` | Unit, integration, acceptance tests |
| `scripts/` | Harness CI scripts |

---

## LLM context files

- [llms.txt](llms.txt) — concise project map for LLM crawlers
- [llms-full.txt](llms-full.txt) — extended context map
- [docs/AGENT_CREATION_GUIDE.md § Instructions for LLM coding agents](docs/AGENT_CREATION_GUIDE.md) — detailed agent instructions

---

## Contact & security

- Maintainer: Artur Czarnecki — jakbu.czarnecki.83@gmail.com
- Security issues: see [SECURITY.md](SECURITY.md)
- Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md)
