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
**Source:** [docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

**Documentation boundary:** `docs/intergrax_runtime_architecture.md` (sole file in `docs/` root) indexes **21 domain pairs**: `docs/architecture/<DOMAIN>.md` ↔ `docs/plan/<DOMAIN>.md` (1:1 filenames). Strategy, ideal model, and audit live in `docs/guides/`. Each **business environment** (`applications/<product>/`) and **business agent** (`agents/<name>/`) has its own architecture and implementation plan — do not treat platform canon as the product deployment plan.

**Per-iteration reading rule:** when implementing a harness layer, read **only** the matching architecture + plan pair (e.g. `MEMORY.md` in both folders) plus `docs/guides/` as needed — do not load unrelated domain docs.

---

## Before you write code

1. Read [README.md — Start here](README.md#start-here) for documentation navigation
2. Read [docs/intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) — pick your domain pair from the table
3. Read **both** `docs/architecture/<DOMAIN>.md` and `docs/plan/<DOMAIN>.md` for that domain only
4. Follow the work cycle in [docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md):

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
- Canonical workflow: [docs/guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md)
- Success metric: idea → first Nexus run in **under one hour**

### Documentation

- **One source of truth per topic** — `docs/` root = hub only; no parallel guides
- Strategy / ideal / audit → `docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`, `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`
- Architecture hub → `docs/intergrax_runtime_architecture.md`
- Domain pairs → `docs/architecture/<DOMAIN>.md` ↔ `docs/plan/<DOMAIN>.md` (**1:1**, same filename)
- Global ladder, DoD, product backlog → `docs/plan/PLATFORM_FOUNDATION.md`
- Agent workflow → `docs/guides/AGENT_CREATION_GUIDE.md`
- Harness AI terms → `docs/architecture/PLATFORM_FOUNDATION.md` §5.3 only
- Nexus execution flow → `docs/architecture/NEXUS_EXECUTION_FLOW.md` + `docs/plan/NEXUS_EXECUTION_FLOW.md` · ADR → `docs/adr/ADR-FLOW-001.md`

### Harness platform

- Default queue is **gate maintenance** in `docs/plan/PLATFORM_FOUNDATION.md` unless another domain plan item is selected
- Business agents (Phase K) are **end of plan** — `docs/plan/PLATFORM_FOUNDATION.md` §6.3; do not start without explicit product decision
- Tier-1/2/3 work is **composition and wiring** of existing Tier-0 modules — no parallel universal mechanisms

---

## Task routing — what to read

| Task | Read first (architecture + plan pair) |
|------|---------------------------------------|
| Create a new agent | [docs/guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) |
| Wire integrations | [INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [plan/INTEGRATIONS.md](docs/plan/INTEGRATIONS.md) |
| RAG / retrieval engine | [RAG.md](docs/architecture/RAG.md) · [plan/RAG.md](docs/plan/RAG.md) |
| Add or use tools | [TOOLS.md](docs/architecture/TOOLS.md) · [plan/TOOLS.md](docs/plan/TOOLS.md) · `intergrax/tools/USAGE.md` |
| Ephemeral Code Craft (dynamic codegen) | [CODE_CRAFT.md](docs/architecture/CODE_CRAFT.md) · [plan/CODE_CRAFT.md](docs/plan/CODE_CRAFT.md) |
| Add or use skills | [SKILLS.md](docs/architecture/SKILLS.md) · [plan/SKILLS.md](docs/plan/SKILLS.md) |
| Configure LLM providers | [LLM_ADAPTERS.md](docs/architecture/LLM_ADAPTERS.md) · [plan/LLM_ADAPTERS.md](docs/plan/LLM_ADAPTERS.md) |
| Memory / context / LTM | [MEMORY.md](docs/architecture/MEMORY.md) · [plan/MEMORY.md](docs/plan/MEMORY.md) |
| New application (Tier-3) | [TIER3_APPLICATION_ENVIRONMENT.md](docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md) · [plan/TIER3_APPLICATION_ENVIRONMENT.md](docs/plan/TIER3_APPLICATION_ENVIRONMENT.md) |
| Plugin / extension | [EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md) · [plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| Governance / policy / UAEP | [UNIFIED_EXECUTION_RUNTIME.md](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [plan/UNIFIED_EXECUTION_RUNTIME.md](docs/plan/UNIFIED_EXECUTION_RUNTIME.md) |
| Orchestration / graphs | [ORCHESTRATION.md](docs/architecture/ORCHESTRATION.md) · [plan/ORCHESTRATION.md](docs/plan/ORCHESTRATION.md) |
| Reasoning / planning / cognition | [REASONING_AND_COGNITION.md](docs/architecture/REASONING_AND_COGNITION.md) · [plan/REASONING_AND_COGNITION.md](docs/plan/REASONING_AND_COGNITION.md) |
| Nexus execution flow | [NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) · [plan/NEXUS_EXECUTION_FLOW.md](docs/plan/NEXUS_EXECUTION_FLOW.md) |
| Agents / registry / capabilities | [AGENT_CONTRACTS_AND_ASSEMBLY.md](docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [plan/AGENT_CONTRACTS_AND_ASSEMBLY.md](docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| Observability | [OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) · [plan/OBSERVABILITY.md](docs/plan/OBSERVABILITY.md) · [ADR-OBS-001](docs/adr/ADR-OBS-001.md) |
| Reliability / HITL | [RELIABILITY_FAILURE_AND_HITL.md](docs/architecture/RELIABILITY_FAILURE_AND_HITL.md) · [plan/RELIABILITY_FAILURE_AND_HITL.md](docs/plan/RELIABILITY_FAILURE_AND_HITL.md) |
| L4 adaptive harness | [ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · [plan/ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| Elastic capacity / platform scaling | [ELASTIC_CAPACITY_AND_SCALING.md](docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md) · [plan/ELASTIC_CAPACITY_AND_SCALING.md](docs/plan/ELASTIC_CAPACITY_AND_SCALING.md) |
| Critic / verification | [CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) · [plan/CRITIC_VERIFICATION.md](docs/plan/CRITIC_VERIFICATION.md) |
| DX / evaluation / gates | [EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](docs/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) · [plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| Platform ladder / product backlog | [PLATFORM_FOUNDATION.md](docs/architecture/PLATFORM_FOUNDATION.md) · [plan/PLATFORM_FOUNDATION.md](docs/plan/PLATFORM_FOUNDATION.md) |
| Available agents (roster) | [agents/README.md](agents/README.md) |
| Available application environments | [applications/README.md](applications/README.md) |
| Harness audit (32 layers) | [docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md](docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md) |

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
python scripts/check_docs_domain_pairs.py
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
| `docs/intergrax_runtime_architecture.md` | Sole `docs/` root file — hub indexing 21 domain pairs |
| `docs/architecture/` | Domain architecture canon (21 files) |
| `docs/plan/` | Domain implementation plans (21 files, 1:1 with architecture) |
| `docs/guides/` | Strategy, ideal model, audit map, authoring guides |
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
- [docs/guides/AGENT_CREATION_GUIDE.md § Instructions for LLM coding agents](docs/guides/AGENT_CREATION_GUIDE.md) — detailed agent instructions

---

## Contact & security

- Maintainer: Artur Czarnecki — jakbu.czarnecki.83@gmail.com
- Security issues: see [SECURITY.md](SECURITY.md)
- Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md)
