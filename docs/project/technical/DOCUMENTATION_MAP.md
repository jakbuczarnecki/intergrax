# Documentation map

**Purpose:** Single navigation hub for Intergrax docs — *what to read, when, and what each artifact is for*.
This file does **not** duplicate canon content; it routes to the authoritative source per topic.

**Audiences:** new developers · Cursor operators · AI coding agents.

Public readers looking for product value, proof paths, evaluation, partnership or license routes should start with [PUBLIC_DOCUMENTATION_MAP.md](../community/PUBLIC_DOCUMENTATION_MAP.md).

---

## Quick routing

| You need… | Read |
|-----------|------|
| Project overview and quick start | [README.md](../../../README.md) |
| **Platform configuration** (env, LLM, embeddings) | [PLATFORM_CONFIGURATION.md](guides/PLATFORM_CONFIGURATION.md) |
| **Public documentation map** (reader routes, proof paths) | [PUBLIC_DOCUMENTATION_MAP.md](../community/PUBLIC_DOCUMENTATION_MAP.md) |
| **Public documentation architecture** (maintainer contract) | [public-adoption/PUBLIC_DOCUMENTATION_ARCHITECTURE.md](../maintainers/public-adoption/PUBLIC_DOCUMENTATION_ARCHITECTURE.md) |
| **Governed Execution** (policy enforcement capability) | [GOVERNED_EXECUTION.md](../architecture/GOVERNED_EXECUTION.md) |
| **Token Optimization main guide** | [capabilities/token_optimization/README.md](../capabilities/token_optimization/README.md) |
| **This map** (roles and workflows) | `docs/project/technical/DOCUMENTATION_MAP.md` |
| **Extend Intergrax / build plugins** | [EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md) → surface matrix → domain guide · design: [PLATFORM_PLUGINS.md](../architecture/PLATFORM_PLUGINS.md) |
| Architecture hub + 24 domain pairs | [intergrax_runtime_architecture.md](../architecture/intergrax_runtime_architecture.md) |
| Multi-layer capability docs | [capabilities/README.md](../capabilities/README.md) — includes `TOKEN_OPTIMIZATION`, `LANGCHAIN_INDEPENDENCE` |
| Domain architecture canon | `docs/project/architecture/<DOMAIN>.md` |
| Implementation status / backlog | `docs/project/maintainers/plans/<DOMAIN>.md` |
| Strategy, invariants, authoring guides | [guides/README.md](guides/README.md) |
| **Documentation Design System** (domain / feature hub standard) | [DOCUMENTATION_DESIGN_SYSTEM.md](guides/DOCUMENTATION_DESIGN_SYSTEM.md) |
| Work with Cursor (AI agent) | [AGENTS.md](../../../AGENTS.md) + [AGENT_INSTRUCTIONS.md](guides/AGENT_INSTRUCTIONS.md) |
| Cursor token budget (F2 / F3) | [CURSOR_TOKEN_SETUP.md](guides/CURSOR_TOKEN_SETUP.md) |
| Audit procedure | [audit/README.md](../maintainers/audit/README.md) |
| Audit session paste (first chat message) | [bootstrap/README.md](../maintainers/bootstrap/README.md) |
| Audit run artifacts | [audit_results/](../../audit_results/) — load only with `RESUME:` |
| Layer closeout (LCM 1–6) | [LAYER_COMPLETION_MODE.md](guides/LAYER_COMPLETION_MODE.md) |
| Milestone narrative log | [implementation-journal/README.md](../maintainers/implementation-journal/README.md) |
| Architectural decisions | [adr/README.md](adr/README.md) |
| **Governed external execution** (ownership, lifecycle, invariants) | [platform/governed_external_execution.md](platform/governed_external_execution.md) |
| **Partner validation readiness** (GEC / ImpeachmentRight five-point matrix) | [integrations/impeachmentright_validation_readiness.md](../integrations/impeachmentright_validation_readiness.md) |
| Contributing / PR process | [CONTRIBUTING.md](../../../CONTRIBUTING.md) |

Domain pair index (24 names): [architecture hub § Domain pair index](../architecture/intergrax_runtime_architecture.md#domain-pair-index-24).

---

## Document roles (what each artifact is)

| Artifact | Role | Not |
|----------|------|-----|
| [README.md](../../../README.md) | Human-facing project intro, maturity snapshot, extended doc index | Full Cursor workflow reference |
| [PUBLIC_DOCUMENTATION_MAP.md](../community/PUBLIC_DOCUMENTATION_MAP.md) | Public reader navigation — intent routes, proof paths, maturity boundary | Technical domain pairs, Cursor workflow |
| [public-adoption/PUBLIC_DOCUMENTATION_ARCHITECTURE.md](../maintainers/public-adoption/PUBLIC_DOCUMENTATION_ARCHITECTURE.md) | Maintainer-owned public information architecture | Implementation plans, detailed claims |
| [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md) | Navigation only — Szukasz → Czytaj | Canon, plan rows, audit prompts |
| [intergrax_runtime_architecture.md](../architecture/intergrax_runtime_architecture.md) | Architecture hub indexing 24 architecture ↔ plan pairs | Per-domain deep spec (use pair files) |
| `docs/project/architecture/<DOMAIN>.md` | **What** the harness should do (contracts, design) | Implementation tracker |
| `docs/project/maintainers/plans/<DOMAIN>.md` | **What is done / next** (phases, rows, gates) | Architecture spec |
| `docs/project/capabilities/architecture/<FEATURE>.md` | **Cross-layer feature architecture** — coordinates domain pairs | Domain canon replacement |
| `docs/project/capabilities/plan/<FEATURE>.md` | **Cross-layer feature plan** — phases across domains; domain rows stay in owning `docs/project/maintainers/plans/<DOMAIN>.md` | Standalone domain plan |
| `docs/project/capabilities/architecture/satellites/` · `docs/project/capabilities/plan/satellites/` | Feature satellite registers (`.cursorignore`; explicit `@` / `Read` only) | Feature root `satellites/` |
| [guides/](guides/README.md) | Cross-cutting strategy, invariants, authoring, audit methodology | Domain canon |
| [DOCUMENTATION_DESIGN_SYSTEM.md](guides/DOCUMENTATION_DESIGN_SYSTEM.md) | **Hub authoring standard** — front section, visuals, gates, maintenance for `architecture/<DOMAIN>.md` and feature hubs | Domain content, plan rows |
| [platform/governed_external_execution.md](platform/governed_external_execution.md) | Cross-cutting platform capability: governed external execution (ownership · lifecycle · invariants) | ADR text, GEC product trackers |
| [integrations/impeachmentright_validation_readiness.md](../integrations/impeachmentright_validation_readiness.md) | Partner-facing readiness / five-point compatibility matrix for governed external execution | Platform architecture canon, attestation/EBE design |
| [AGENTS.md](../../../AGENTS.md) | Cursor auto-load **stub** (~350 tok) | Full agent instructions |
| [AGENT_INSTRUCTIONS.md](guides/AGENT_INSTRUCTIONS.md) | Full AI agent reference (routing, verification, ADR, O1 output) | Human onboarding doc |
| [.cursor/rules/](../../../.cursor/rules/README.md) | Always-on / triggered Cursor rules | Replacement for AGENT_INSTRUCTIONS |
| [bootstrap/*.txt](../maintainers/bootstrap/README.md) | Copy-paste **first message** for a new agent chat | Stored audit results |
| [audit/<DOMAIN>.md](../maintainers/audit/README.md) | Per-domain audit prompts (generated) | Implementation plan |
| [audit_results/](../../audit_results/) | Run output (`progress.json`, reports) | Load in Cursor unless `RESUME:` cites path |
| [LAYER_COMPLETION_MODE.md](guides/LAYER_COMPLETION_MODE.md) | **When/how** deep domain closeout (LCM steps) | Bootstrap paste file |
| `docs/project/maintainers/audit/*_ORCHESTRATOR.md` | Mode-specific procedure (A / B / C / I) | General onboarding |
| [implementation-journal/](../maintainers/implementation-journal/README.md) | Milestone narrative (optional) | Plan source of truth or ADR store |

**One source of truth per topic.** Canonical platform and cross-cutting documentation lives under `docs/project/`. Application-owned technical canon lives under `applications/*/docs/` (architecture, implementation plan, build/deploy, ADRs, application evidence). Agent-owned technical canon lives under `agents/*/docs/` (architecture, implementation plan, ADRs). Code-local README files and workflow-adjacent artifacts may remain at the application or agent root when required by tooling; those are not competing documentation roots.

---

## By audience

### New developer (human)

1. [README.md](../../../README.md) — overview, install, verify
2. [intergrax_runtime_architecture.md](../architecture/intergrax_runtime_architecture.md) — pick a domain
3. Domain-layer pair: `docs/project/architecture/<DOMAIN>.md` + `docs/project/maintainers/plans/<DOMAIN>.md`
4. Cross-layer capabilities (when relevant): [capabilities/README.md](../capabilities/README.md) — `docs/project/capabilities/architecture/<FEATURE>.md` ↔ `docs/project/capabilities/plan/<FEATURE>.md`
5. [SYSTEM_INVARIANTS.md](guides/SYSTEM_INVARIANTS.md) before changing code
6. [CONTRIBUTING.md](../../../CONTRIBUTING.md) for PR workflow

Authoring: [AGENT_CREATION_GUIDE.md](guides/AGENT_CREATION_GUIDE.md) · Tier-3: [applications/USAGE.md](../../../applications/USAGE.md)

**Extend Intergrax (plugins):** [EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md) — surface decision tree + 12-surface matrix → domain author guide or architecture pair. System design: [PLATFORM_PLUGINS.md](../architecture/PLATFORM_PLUGINS.md). Maintainer audit/roadmap: [PLATFORM_PLUGIN_DOCUMENTATION_AUDIT.md](../maintainers/plans/PLATFORM_PLUGIN_DOCUMENTATION_AUDIT.md) (not the first stop for plugin authors).

### Cursor operator (audit / implement / closeout)

```text
bootstrap paste  →  audit orchestrator  →  audit_results/
     ↑                      ↑
docs/project/maintainers/bootstrap/      docs/project/maintainers/audit/README.md
```

| Goal | Start here |
|------|------------|
| Audit platform (no code) | [bootstrap/02_audit_one_domain.txt](../maintainers/bootstrap/02_audit_one_domain.txt) or [06](../maintainers/bootstrap/06_interactive_layer_by_layer_audit.txt) |
| Implement open plan items | [bootstrap/04_implement_plan_one_domain.txt](../maintainers/bootstrap/04_implement_plan_one_domain.txt) |
| Full layer closeout (LCM) | [bootstrap/05_closeout_all_domains.txt](../maintainers/bootstrap/05_closeout_all_domains.txt) + [LAYER_COMPLETION_MODE.md](guides/LAYER_COMPLETION_MODE.md) |
| New idea before build (Mode I) | Natural language in new chat — see [IDEA_AUDIT_ORCHESTRATOR.md](../maintainers/audit/IDEA_AUDIT_ORCHESTRATOR.md) |
| CI preflight before push | [bootstrap/07_ci_preflight.txt](../maintainers/bootstrap/07_ci_preflight.txt) |

**F3:** one domain = one new chat. **Resume:** same bootstrap + `RESUME: docs/audit_results/YYYY-MM-DD/progress.json`

### AI coding agent (Cursor)

```text
AGENTS.md (stub)  →  AGENT_INSTRUCTIONS.md  →  .cursor/rules/intergrax-iteration.mdc
```

- Load domain canon on demand — not bulk guides (respect `.cursorignore`)
- Audit context: [audit_slices/<DOMAIN>.md](guides/audit_slices) — not full audit prompts unless auditing
- Default scope: gate maintenance in [plan/PLATFORM_FOUNDATION.md](../maintainers/plans/PLATFORM_FOUNDATION.md) hub read-scope

---

## What not to load by default

| Path / pattern | Why |
|----------------|-----|
| `.cursorignore` entries | Token budget; grep `SYS-INV-*` or `@` explicit |
| Full multi-thousand-line plan hubs | Use hub read-scope + one satellite |
| `docs/audit_results/` | Run artifacts — only with operator `RESUME:` line |
| Unrelated domain pairs | F3: one domain per session |
| [llms-full.txt](../../../llms-full.txt) | Bulk context; prefer hub + pair |

---

## Workflow overview

```text
                    DOCUMENTATION_MAP.md
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
    Human dev          Cursor op           AI agent
    README             bootstrap           AGENTS.md
    CONTRIBUTING       audit/README        AGENT_INSTRUCTIONS
         │                  │                  │
         └──────────►  architecture ↔ plan  ◄─┘
                    (24 domain pairs via hub)
                    capabilities/architecture ↔ capabilities/plan
                    (multi-layer feature pairs)
```

**Update rule:** When adding a new doc class, add one row to *Quick routing* and *Document roles* here — do not duplicate content in README beyond a short pointer.
