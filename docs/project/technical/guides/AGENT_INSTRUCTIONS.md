# Intergrax — Instructions for AI Coding Agents (full reference)

> **Audience:** Cursor, Claude Code, Codex, Gemini, and other LLM agents working in this repository.
> **Cursor auto-load:** root [`AGENTS.md`](../../../../AGENTS.md) is a **stub** (~400 tokens). Load **this file** with `@docs/project/technical/guides/AGENT_INSTRUCTIONS.md` when you need routing, verification, ADR workflow, or anti-patterns.
> **Canonical docs:** `docs/project/` — always prefer linking over duplicating.

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
**Source:** [docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md)

**Documentation boundary:** `docs/project/architecture/intergrax_runtime_architecture.md` is the architecture hub indexing **22 domain-layer pairs**: `docs/project/architecture/<DOMAIN>.md` ↔ `docs/project/maintainers/plans/<DOMAIN>.md` (1:1 filenames). **Multi-layer feature pairs** live under `docs/project/capabilities/architecture/<FEATURE>.md` ↔ `docs/project/capabilities/plan/<FEATURE>.md` — see [`capabilities/README.md`](../../capabilities/README.md). Feature docs coordinate cross-layer capabilities; domain docs remain authoritative for domain-owned architecture and plan rows. Strategy, ideal model, and audit live in `docs/project/technical/guides/`. Each **business environment** (`applications/<product>/`) and **business agent** (`agents/<name>/`) has its own architecture and implementation plan — do not treat platform canon as the product deployment plan.

**Per-iteration reading rule:** when implementing a harness layer, read **only** the matching architecture + plan pair (e.g. `MEMORY.md` in both folders) plus `docs/project/technical/guides/` as needed — do not load unrelated domain docs.

**Cursor context budget:** respect `.cursorignore`. **I1/O1:** always-on `.cursor/rules/intergrax-token-budget.mdc`. Plan hubs + [`../plan/satellites/`](../../maintainers/plans/satellites) satellites. Audits: [`audit_slices/<DOMAIN>.md`](audit_slices). **F2:** root `AGENTS.md` is a stub; full reference is this file — see [`CURSOR_TOKEN_SETUP.md`](CURSOR_TOKEN_SETUP.md). **F3:** one domain = one new chat; HEP → [`../bootstrap/hep_step.txt`](../../maintainers/bootstrap/hep_step.txt). **O1:** terse operator replies by default — see § Operator communication below.

---

## Operator communication (O1 — output token budget)

**Minimize output tokens.** Do not dump architecture canon, repeat visible diffs, or end with unsolicited long “next steps” lists.

### Response modes

| Mode | When | Shape |
|------|------|--------|
| **Minimal** | Operator: `krótko`, `terse`; trivial yes/no | ≤6 lines |
| **Terse** | Default — implement, fix, gate, routine audit checkpoint | ≤12 lines (~150 words) |
| **Standard** | Operator: `wyjaśnij`, `explain`, design review | Short sections; link to docs instead of quoting |
| **Full** | `pełny raport`, `full report`, `iteration summary`; milestone / LCM / journal entry | 12-point template below |

Language: operator session language for chat; repository artifacts stay English.

### Terse default (unless Full triggered)

Include **only**:

1. **Outcome** — done / blocked / partial (+ one-line why if not obvious)
2. **Changed** — file paths (or count if >5); never narrate the diff
3. **Tests** — command + pass/fail, or one line why skipped
4. **Next** — one line max; omit if nothing needed

**Skip:** preamble, restating the task, code blocks for unchanged context, tables duplicating CI output, Mode I–style long proposals when not in Mode I.

### Full iteration summary (on request or milestone only)

1. Completed implementation item
2. Domain pair (`architecture/<DOMAIN>.md` + `plan/<DOMAIN>.md`) and Harness layer
3. Changed files
4. Tests added or updated
5. Tests executed (commands + result)
6. Documentation updated (domain pair files)
7. Architectural impact
8. Remaining risks
9. Out-of-scope findings
10. Suggested next step (one line)
11. One-line commit message (English) — no commit unless operator asks
12. Journal — entry path **only if written**; else **"no journal needed"** + one-line rationale

---

## Before you write code

**CI/test hotfixes:** use `@.cursor/rules/intergrax-ci-hotfix.mdc` in a **new chat**. Do **not** read README, architecture, plan, `SYSTEM_INVARIANTS.md`, or strategy docs. Read only failing test/checker + directly related implementation files.

**Full onboarding** — only for new domain implementation, architecture-changing work, public behavior changes, ADR/milestone work, or full audit/closeout:

1. Read [README.md — Start here](../../../../README.md#start-here) for documentation navigation
2. Read [docs/project/architecture/intergrax_runtime_architecture.md](../../architecture/intergrax_runtime_architecture.md) — pick your domain pair from the table
3. Skim [docs/project/technical/guides/SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) — cross-domain rules you must not break (P2-ARCH-01)
4. Read **both** `docs/project/architecture/<DOMAIN>.md` and `docs/project/maintainers/plans/<DOMAIN>.md` for that domain only
5. Follow the work cycle in [docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md):

```text
ANALYSIS → ARCHITECTURE REVIEW → PLAN REVIEW → IMPROVEMENT PROPOSAL
  → DOCUMENTATION UPDATE → IMPLEMENTATION → VERIFICATION → CONCLUSIONS
```

---

## Hard rules (never violate)

Full cross-domain index: [docs/project/technical/guides/SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) (`SYS-INV-*`, P2-ARCH-01). Summary below — when in doubt, use the index and linked domain canon.

### Tier dependency boundaries

```text
intergrax/       MUST NOT import from agents/ or applications/
agents/          MUST NOT import from applications/
applications/    MAY import from agents/ and intergrax/
```

### Agent creation

- **Never modify `../../intergrax/runtime/`** when creating Tier-2 agents
- Agents consume Tier-0 only through Nexus policy and `ToolRuntime` — no direct vendor SDK imports
- Canonical workflow: [docs/project/technical/guides/AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md)
- Success metric: idea → first Nexus run in **under one hour**

### Documentation

- **One source of truth per topic** — `docs/project/` is the canonical human documentation root; no parallel guides
- Strategy / ideal / audit / invariants → `docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`, `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, `SYSTEM_INVARIANTS.md`
- Architecture hub → `docs/project/architecture/intergrax_runtime_architecture.md`
- Domain-layer pairs → `docs/project/architecture/<DOMAIN>.md` ↔ `docs/project/maintainers/plans/<DOMAIN>.md` (**1:1**, same filename)
- Multi-layer feature pairs → `docs/project/capabilities/architecture/<FEATURE>.md` ↔ `docs/project/capabilities/plan/<FEATURE>.md` (**1:1**; do not create `docs/project/maintainers/plans/<FEATURE>.md` for cross-layer features)
- Global ladder, DoD, product backlog → `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md`
- Agent workflow → `docs/project/technical/guides/AGENT_CREATION_GUIDE.md`
- Harness AI terms → `docs/project/architecture/PLATFORM_FOUNDATION.md` §5.3 only
- Nexus execution flow → `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` + `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` · ADR → `docs/project/technical/adr/entries/2026-06-07/ADR-FLOW-001.md`
- Completed implementation **milestones** → `docs/project/maintainers/implementation-journal/` (optional for routine iterations; see journal README)

### Harness platform

- Default queue is **gate maintenance** in `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` unless another domain plan item is selected
- Business agents (Phase K) are **end of plan** — `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` §6.3; do not start without explicit product decision
- Tier-1/2/3 work is **composition and wiring** of existing Tier-0 modules — no parallel universal mechanisms

---

## Task routing — what to read

| Task | Read first (architecture + plan pair) |
|------|---------------------------------------|
| Audit a new idea before build | Say `Zrób audyt pomysłu: …` in a new chat — rule `.cursor/rules/intergrax-idea-audit.mdc` → [`idea_audit.txt`](../../maintainers/bootstrap/idea_audit.txt) · [`IDEA_AUDIT_ORCHESTRATOR.md`](../../maintainers/audit/IDEA_AUDIT_ORCHESTRATOR.md) |
| Cross-layer platform feature | [`capabilities/README.md`](../../capabilities/README.md) → matching feature architecture + feature plan, then affected domain pairs |
| Create a new agent | [docs/project/technical/guides/AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) |
| Wire integrations | [INTEGRATIONS.md](../../architecture/INTEGRATIONS.md) · [plan/INTEGRATIONS.md](../../maintainers/plans/INTEGRATIONS.md) |
| RAG / retrieval engine | [RAG.md](../../architecture/RAG.md) · [plan/RAG.md](../../maintainers/plans/RAG.md) |
| Add or use tools | [TOOLS.md](../../architecture/TOOLS.md) · [plan/TOOLS.md](../../maintainers/plans/TOOLS.md) · `../../intergrax/tools/USAGE.md` |
| Ephemeral Code Craft (dynamic codegen) | [CODE_CRAFT.md](../../architecture/CODE_CRAFT.md) · [plan/CODE_CRAFT.md](../../maintainers/plans/CODE_CRAFT.md) |
| Add or use skills | [SKILLS.md](../../architecture/SKILLS.md) · [plan/SKILLS.md](../../maintainers/plans/SKILLS.md) |
| Configure LLM providers | [LLM_ADAPTERS.md](../../architecture/LLM_ADAPTERS.md) · [plan/LLM_ADAPTERS.md](../../maintainers/plans/LLM_ADAPTERS.md) |
| Memory / LTM stores | [MEMORY.md](../../architecture/MEMORY.md) · [plan/MEMORY.md](../../maintainers/plans/MEMORY.md) |
| Context engineering engine | [CONTEXT_ENGINEERING.md](../../architecture/CONTEXT_ENGINEERING.md) · [plan/CONTEXT_ENGINEERING.md](../../maintainers/plans/CONTEXT_ENGINEERING.md) |
| New application (Tier-3) | [APPLICATION_CREATION_GUIDE.md](APPLICATION_CREATION_GUIDE.md) · [TIER3_APPLICATION_ENVIRONMENT.md](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) |
| Plugin / extension | [EXTENSION_AUTHOR_GUIDE.md](EXTENSION_AUTHOR_GUIDE.md) · [plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](../../maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| Governance / policy / UAEP | [UNIFIED_EXECUTION_RUNTIME.md](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) · [plan/UNIFIED_EXECUTION_RUNTIME.md](../../maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md) |
| Orchestration / graphs | [ORCHESTRATION.md](../../architecture/ORCHESTRATION.md) · [plan/ORCHESTRATION.md](../../maintainers/plans/ORCHESTRATION.md) |
| Reasoning / planning / cognition | [REASONING_AND_COGNITION.md](../../architecture/REASONING_AND_COGNITION.md) · [plan/REASONING_AND_COGNITION.md](../../maintainers/plans/REASONING_AND_COGNITION.md) |
| Nexus execution flow | [NEXUS_EXECUTION_FLOW.md](../../architecture/NEXUS_EXECUTION_FLOW.md) · [plan/NEXUS_EXECUTION_FLOW.md](../../maintainers/plans/NEXUS_EXECUTION_FLOW.md) |
| Agents / registry / capabilities | [AGENT_CONTRACTS_AND_ASSEMBLY.md](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [plan/AGENT_CONTRACTS_AND_ASSEMBLY.md](../../maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| Observability | [OBSERVABILITY.md](../../architecture/OBSERVABILITY.md) · [plan/OBSERVABILITY.md](../../maintainers/plans/OBSERVABILITY.md) · [ADR-OBS-001](../adr/entries/2026-06-08/ADR-OBS-001.md) |
| Reliability / HITL | [RELIABILITY_FAILURE_AND_HITL.md](../../architecture/RELIABILITY_FAILURE_AND_HITL.md) · [plan/RELIABILITY_FAILURE_AND_HITL.md](../../maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md) |
| L4 adaptive harness | [ADAPTIVE_HARNESS_INTELLIGENCE.md](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · [plan/ADAPTIVE_HARNESS_INTELLIGENCE.md](../../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| Elastic capacity / platform scaling | [ELASTIC_CAPACITY_AND_SCALING.md](../../architecture/ELASTIC_CAPACITY_AND_SCALING.md) · [plan/ELASTIC_CAPACITY_AND_SCALING.md](../../maintainers/plans/ELASTIC_CAPACITY_AND_SCALING.md) |
| Critic / verification | [CRITIC_VERIFICATION.md](../../architecture/CRITIC_VERIFICATION.md) · [plan/CRITIC_VERIFICATION.md](../../maintainers/plans/CRITIC_VERIFICATION.md) |
| DX / evaluation / gates | [EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) · [plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](../../maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| Platform ladder / product backlog | [PLATFORM_FOUNDATION.md](../../architecture/PLATFORM_FOUNDATION.md) · [plan/PLATFORM_FOUNDATION.md](../../maintainers/plans/PLATFORM_FOUNDATION.md) |
| Available agents (roster) | [agents/README.md](../../../../agents/README.md) |
| Available application environments | [applications/README.md](../../../../applications/README.md) |
| Harness audit (32 layers) | [docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md](INTEGRAX_HARNESS_AUDIT_MAP.md) |
| Architecture audit orchestration (22 pairs) | [docs/project/maintainers/audit/README.md](../../maintainers/audit/README.md) · [docs/project/maintainers/bootstrap/](../../maintainers/bootstrap/README.md) · `scripts/audit/init_architecture_audit_run.py` |
| System invariants (never violate) | [docs/project/technical/guides/SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) |
| Layer completion (full domain closeout) | [docs/project/technical/guides/LAYER_COMPLETION_MODE.md](LAYER_COMPLETION_MODE.md) |
| Implementation journal | [docs/project/maintainers/implementation-journal/README.md](../../maintainers/implementation-journal/README.md) |

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

**CI/test hotfix:** run only the failing command from GitHub, the exact failing test file, or `uv run python scripts/ci/run_ci_smoke_pytest.py`. Do **not** run the full checklist below unless the operator explicitly requests full harness verification.

**Full harness verification** (domain implementation, architecture changes, milestone closeout):

```bash
uv run pytest -m "gate and not no_ci" -q
python scripts/maintenance/check_harness_no_getattr.py
uv run python scripts/maintenance/check_observability_gates.py
python scripts/audit/check_docs_domain_pairs.py
python scripts/audit/check_idea_audit_bootstrap.py
python scripts/maintenance/check_reasoning_gates.py
python scripts/maintenance/check_implementation_journal.py
python scripts/maintenance/check_harness_adr.py
python scripts/maintenance/check_plan_hub_size.py
python scripts/ci/check_cursor_token_setup.py
python scripts/maintenance/check_arch_hub_size.py
python scripts/audit/check_token_generator_freshness.py
python scripts/audit/check_audit_token_discipline.py
uv run python scripts/gates/check_agent_acp_close_ci.py
python scripts/maintenance/check_production_capacity_adapters.py
python scripts/maintenance/check_harness_resilience_policy.py
```

For agent-only work:

```bash
uv run pytest agents/<agent>/tests/ -q
```

Full local suite: `scripts\ci\test.bat unit` (Windows) or equivalent `uv run pytest`.

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
- Importing `agents/` or `applications/` from `../../intergrax/`
- Direct vendor SDK usage in Tier-2 agents
- Modifying Nexus runtime for agent-specific needs
- Creating new universal mechanisms when Tier-0 already provides one
- Starting Phase K business agents without explicit product prioritization
- Committing secrets (`.env`, credentials, API keys)

---

## Key paths

| Path | Contents |
|------|----------|
| `docs/project/architecture/intergrax_runtime_architecture.md` | Architecture hub indexing 22 domain pairs + feature doc index |
| `docs/project/architecture/` | Domain architecture canon (22 files) |
| `docs/project/maintainers/plans/` | Domain implementation plans (22 files, 1:1 with architecture) |
| `docs/project/capabilities/` | Multi-layer feature architecture + plan pairs (1:1 under `architecture/` and `plan/`) |
| `docs/project/technical/guides/` | Strategy, ideal model, audit map, authoring guides |
| `../../intergrax/runtime/nexus/` | Nexus Agent OS core |
| `../../intergrax/runtime/nexus/orchestration/` | Intake, planning, graph, HITL runners |
| `../../intergrax/integrations/` | Integration Library |
| `../../intergrax/tools/` | Tool Library |
| `../../intergrax/skills/` | Skill Library |
| `../../intergrax/llm_adapters/` | LLM provider adapters |
| `../../intergrax/rag/` | RAG engine |
| `../../intergrax/scaffold/` | Scaffolding CLI |
| `agents/` | Tier-2 agents — roster: [agents/README.md](../../../../agents/README.md) |
| `applications/` | Tier-3 application hosts — index: [applications/README.md](../../../../applications/README.md) (LKW, DSW, legal, research, lab) |
| `docs/project/` | Canonical human documentation |
| `tests/` | Unit, integration, acceptance tests |
| `scripts/` | Harness CI scripts |

---

## LLM context files

- [llms.txt](../../../../llms.txt) — concise project map for LLM crawlers
- [llms-full.txt](../../../../llms-full.txt) — extended context map
- [docs/project/technical/guides/AGENT_CREATION_GUIDE.md § Instructions for LLM coding agents](AGENT_CREATION_GUIDE.md) — detailed agent instructions

---

## Contact & security

- Maintainer: Artur Czarnecki — jakbu.czarnecki.83@gmail.com
- Security issues: see [SECURITY.md](../../../../SECURITY.md)
- Contributing: see [CONTRIBUTING.md](../../../../CONTRIBUTING.md)