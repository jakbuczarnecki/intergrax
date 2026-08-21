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
| Platform audit procedure | [audit_results/README.md](../../audit_results/README.md) |
| **Platform proofs** (methodology, coverage map) | [platform_proofs/README.md](../../../platform_proofs/README.md) |
| Conduct a platform audit | [audit_results/AUDIT_PROTOCOL.md](../../audit_results/AUDIT_PROTOCOL.md) |
| Remediate accepted audit findings | [audit_results/AUDIT_REMEDIATION_PROTOCOL.md](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md) |
| Audit campaigns and historical results | [audit_results/](../../audit_results/) |
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
| [bootstrap/*.txt](../maintainers/bootstrap/README.md) | Copy-paste **first message** for a new agent chat (HEP, CI preflight, micro implement) | Stored audit results |
| [platform_proofs/](../../../platform_proofs/) | **Platform mechanism proofs** — protocol, coverage map, authoring; execution via `scripts/proof/` | Product proofs, public dashboard, duplicate runners |
| [audit_results/](../../audit_results/) | **Canonical** platform audit methodology, campaigns, results, remediation status | Architecture canon or implementation plan |
| [audit_results/AUDIT_PROTOCOL.md](../../audit_results/AUDIT_PROTOCOL.md) | Model-executable adversarial audit instruction | Per-domain generated prompts |
| [audit_results/AUDIT_REMEDIATION_PROTOCOL.md](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md) | Remediation workflow from accepted findings | Duplicate audit narrative in plans |
| [LAYER_COMPLETION_MODE.md](guides/LAYER_COMPLETION_MODE.md) | **When/how** deep domain closeout (LCM steps) | Bootstrap paste file |
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

### Cursor operator (audit / remediate / implement)

| Goal | Start here |
|------|------------|
| Design / run platform proof | [platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md](../../../platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md) |
| Conduct platform audit | [audit_results/AUDIT_PROTOCOL.md](../../audit_results/AUDIT_PROTOCOL.md) |
| Remediate accepted findings | [audit_results/AUDIT_REMEDIATION_PROTOCOL.md](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md) |
| Implement plan / HEP step | [bootstrap/hep_step.txt](../maintainers/bootstrap/hep_step.txt) or [micro_implement.txt](../maintainers/bootstrap/micro_implement.txt) |
| Deep layer closeout (LCM) | [LAYER_COMPLETION_MODE.md](guides/LAYER_COMPLETION_MODE.md) |
| CI preflight before push | [bootstrap/07_ci_preflight.txt](../maintainers/bootstrap/07_ci_preflight.txt) |

**F3:** one domain = one new chat when auditing or implementing a single domain layer.

### AI coding agent (Cursor)

```text
AGENTS.md (stub)  →  AGENT_INSTRUCTIONS.md  →  .cursor/rules/intergrax-iteration.mdc
```

- Load domain canon on demand — not bulk guides (respect `.cursorignore`)
- Platform audit: follow [audit_results/AUDIT_PROTOCOL.md](../../audit_results/AUDIT_PROTOCOL.md) — load campaign paths only when operator cites them
- Default scope: gate maintenance in [plan/PLATFORM_FOUNDATION.md](../maintainers/plans/PLATFORM_FOUNDATION.md) hub read-scope

---

## What not to load by default

| Path / pattern | Why |
|----------------|-----|
| `.cursorignore` entries | Token budget; grep `SYS-INV-*` or `@` explicit |
| Full multi-thousand-line plan hubs | Use hub read-scope + one satellite |
| `docs/audit_results/` | Campaign artifacts — load only when auditing/remediating or operator cites path |
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
    README             audit_results/        AGENTS.md
    CONTRIBUTING       AUDIT_PROTOCOL        AGENT_INSTRUCTIONS
         │                  │                  │
         └──────────►  architecture ↔ plan  ◄─┘
                    (24 domain pairs via hub)
                    capabilities/architecture ↔ capabilities/plan
                    (multi-layer feature pairs)
```

**Update rule:** When adding a new doc class, add one row to *Quick routing* and *Document roles* here — do not duplicate content in README beyond a short pointer.
