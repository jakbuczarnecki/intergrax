# Documentation map

**Purpose:** Single navigation hub for Intergrax docs — *what to read, when, and what each artifact is for*.  
This file does **not** duplicate canon content; it routes to the authoritative source per topic.

**Audiences:** new developers · Cursor operators · AI coding agents.

---

## Quick routing

| You need… | Read |
|-----------|------|
| Project overview and quick start | [README.md](../README.md) |
| **This map** (roles and workflows) | `docs/DOCUMENTATION_MAP.md` |
| Architecture hub + 22 domain pairs | [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) |
| Multi-layer feature docs (cross-layer capabilities) | [features/README.md](features/README.md) — includes `TOKEN_OPTIMIZATION`, `LANGCHAIN_INDEPENDENCE` |
| Domain architecture canon | `docs/architecture/<DOMAIN>.md` |
| Implementation status / backlog | `docs/plan/<DOMAIN>.md` |
| Strategy, invariants, authoring guides | [guides/README.md](guides/README.md) |
| Work with Cursor (AI agent) | [AGENTS.md](../AGENTS.md) + [AGENT_INSTRUCTIONS.md](guides/AGENT_INSTRUCTIONS.md) |
| Cursor token budget (F2 / F3) | [CURSOR_TOKEN_SETUP.md](guides/CURSOR_TOKEN_SETUP.md) |
| Audit procedure | [audit/README.md](audit/README.md) |
| Audit session paste (first chat message) | [bootstrap/README.md](bootstrap/README.md) |
| Audit run artifacts | [audit_results/](audit_results/) — load only with `RESUME:` |
| Layer closeout (LCM 1–6) | [LAYER_COMPLETION_MODE.md](guides/LAYER_COMPLETION_MODE.md) |
| Milestone narrative log | [implementation-journal/README.md](implementation-journal/README.md) |
| Architectural decisions | [adr/README.md](adr/README.md) |
| **Governed external execution** (ownership, lifecycle, invariants) | [platform/governed_external_execution.md](platform/governed_external_execution.md) |
| **Partner validation readiness** (GEC / ImpeachmentRight five-point matrix) | [integrations/impeachmentright_validation_readiness.md](integrations/impeachmentright_validation_readiness.md) |
| Contributing / PR process | [CONTRIBUTING.md](../CONTRIBUTING.md) |

Domain pair index (22 names): [audit/README.md § Domain index](audit/README.md#domain-index-22-pairs) or [architecture hub](intergrax_runtime_architecture.md).

---

## Document roles (what each artifact is)

| Artifact | Role | Not |
|----------|------|-----|
| [README.md](../README.md) | Human-facing project intro, maturity snapshot, extended doc index | Full Cursor workflow reference |
| [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md) | Navigation only — Szukasz → Czytaj | Canon, plan rows, audit prompts |
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) | Sole `docs/` root file; indexes 22 architecture ↔ plan pairs | Per-domain deep spec (use pair files) |
| `docs/architecture/<DOMAIN>.md` | **What** the harness should do (contracts, design) | Implementation tracker |
| `docs/plan/<DOMAIN>.md` | **What is done / next** (phases, rows, gates) | Architecture spec |
| `docs/features/architecture/<FEATURE>.md` | **Cross-layer feature architecture** — coordinates domain pairs | Domain canon replacement |
| `docs/features/plan/<FEATURE>.md` | **Cross-layer feature plan** — phases across domains; domain rows stay in owning `docs/plan/<DOMAIN>.md` | Standalone domain plan |
| `docs/features/architecture/satellites/` · `docs/features/plan/satellites/` | Feature satellite registers (`.cursorignore`; explicit `@` / `Read` only) | Feature root `satellites/` |
| [guides/](guides/README.md) | Cross-cutting strategy, invariants, authoring, audit methodology | Domain canon |
| [platform/governed_external_execution.md](platform/governed_external_execution.md) | Cross-cutting platform capability: governed external execution (ownership · lifecycle · invariants) | ADR text, GEC product trackers |
| [integrations/impeachmentright_validation_readiness.md](integrations/impeachmentright_validation_readiness.md) | Partner-facing readiness / five-point compatibility matrix for governed external execution | Platform architecture canon, attestation/EBE design |
| [AGENTS.md](../AGENTS.md) | Cursor auto-load **stub** (~350 tok) | Full agent instructions |
| [AGENT_INSTRUCTIONS.md](guides/AGENT_INSTRUCTIONS.md) | Full AI agent reference (routing, verification, ADR, O1 output) | Human onboarding doc |
| [.cursor/rules/](../.cursor/rules/README.md) | Always-on / triggered Cursor rules | Replacement for AGENT_INSTRUCTIONS |
| [bootstrap/*.txt](bootstrap/README.md) | Copy-paste **first message** for a new agent chat | Stored audit results |
| [audit/<DOMAIN>.md](audit/README.md) | Per-domain audit prompts (generated) | Implementation plan |
| [audit_results/](audit_results/) | Run output (`progress.json`, reports) | Load in Cursor unless `RESUME:` cites path |
| [LAYER_COMPLETION_MODE.md](guides/LAYER_COMPLETION_MODE.md) | **When/how** deep domain closeout (LCM steps) | Bootstrap paste file |
| `docs/audit/*_ORCHESTRATOR.md` | Mode-specific procedure (A / B / C / I) | General onboarding |
| [implementation-journal/](implementation-journal/README.md) | Milestone narrative (optional) | Plan source of truth or ADR store |

**One source of truth per topic.** Platform canon in `docs/`; product/agent docs under `applications/<name>/` and `agents/<name>/`.

---

## By audience

### New developer (human)

1. [README.md](../README.md) — overview, install, verify  
2. [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) — pick a domain  
3. Domain-layer pair: `docs/architecture/<DOMAIN>.md` + `docs/plan/<DOMAIN>.md`  
4. Cross-layer features (when relevant): [features/README.md](features/README.md) — `docs/features/architecture/<FEATURE>.md` ↔ `docs/features/plan/<FEATURE>.md`  
5. [SYSTEM_INVARIANTS.md](guides/SYSTEM_INVARIANTS.md) before changing code  
6. [CONTRIBUTING.md](../CONTRIBUTING.md) for PR workflow  

Authoring: [AGENT_CREATION_GUIDE.md](guides/AGENT_CREATION_GUIDE.md) · Tier-3: [applications/USAGE.md](../applications/USAGE.md)

### Cursor operator (audit / implement / closeout)

```text
bootstrap paste  →  audit orchestrator  →  audit_results/
     ↑                      ↑
docs/bootstrap/      docs/audit/README.md
```

| Goal | Start here |
|------|------------|
| Audit platform (no code) | [bootstrap/02_audit_one_domain.txt](bootstrap/02_audit_one_domain.txt) or [06](bootstrap/06_interactive_layer_by_layer_audit.txt) |
| Implement open plan items | [bootstrap/04_implement_plan_one_domain.txt](bootstrap/04_implement_plan_one_domain.txt) |
| Full layer closeout (LCM) | [bootstrap/05_closeout_all_domains.txt](bootstrap/05_closeout_all_domains.txt) + [LAYER_COMPLETION_MODE.md](guides/LAYER_COMPLETION_MODE.md) |
| New idea before build (Mode I) | Natural language in new chat — see [IDEA_AUDIT_ORCHESTRATOR.md](audit/IDEA_AUDIT_ORCHESTRATOR.md) |
| CI preflight before push | [bootstrap/07_ci_preflight.txt](bootstrap/07_ci_preflight.txt) |

**F3:** one domain = one new chat. **Resume:** same bootstrap + `RESUME: docs/audit_results/YYYY-MM-DD/progress.json`

### AI coding agent (Cursor)

```text
AGENTS.md (stub)  →  AGENT_INSTRUCTIONS.md  →  .cursor/rules/intergrax-iteration.mdc
```

- Load domain canon on demand — not bulk guides (respect `.cursorignore`)  
- Audit context: [audit_slices/<DOMAIN>.md](guides/audit_slices/) — not full audit prompts unless auditing  
- Default scope: gate maintenance in [plan/PLATFORM_FOUNDATION.md](plan/PLATFORM_FOUNDATION.md) hub read-scope  

---

## What not to load by default

| Path / pattern | Why |
|----------------|-----|
| `.cursorignore` entries | Token budget; grep `SYS-INV-*` or `@` explicit |
| Full multi-thousand-line plan hubs | Use hub read-scope + one satellite |
| `docs/audit_results/` | Run artifacts — only with operator `RESUME:` line |
| Unrelated domain pairs | F3: one domain per session |
| [llms-full.txt](../llms-full.txt) | Bulk context; prefer hub + pair |

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
                    (22 domain pairs via hub)
                    features/architecture ↔ features/plan
                    (multi-layer feature pairs)
```

**Update rule:** When adding a new doc class, add one row to *Quick routing* and *Document roles* here — do not duplicate content in README beyond a short pointer.
