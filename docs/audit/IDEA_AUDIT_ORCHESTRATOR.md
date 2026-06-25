# Idea Audit — Orchestration (Mode I)

**Status:** Canonical orchestrator for **Mode I** — live audit of a **single product or harness idea** before implementation  
**Bootstrap (procedure):** [`bootstrap/idea_audit.txt`](../bootstrap/idea_audit.txt)  
**Cursor rule:** `.cursor/rules/intergrax-idea-audit.mdc` — auto-triggers on "audyt pomysłu" / "idea audit"  
**Related:** [`README.md`](README.md) · [`ORCHESTRATOR.md`](ORCHESTRATOR.md) · [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md)

---

## Purpose

Provide a **repeatable, evidence-backed intake audit** when an operator or developer proposes:

- a new **integration**, **tool**, **skill**, **agent**, or **application**;
- a **harness capability** or cross-cutting platform change;
- an **improvement** or **fix** to existing behavior;
- a **feasibility / architecture analysis** before committing to build.

**One Cursor session** → agent reads the fixed procedure + operator's idea in chat → delivers a structured verdict in chat.

**No idea-audit artifact files** — the durable record after approval is an update to the relevant **domain-layer pair** (`docs/architecture/<DOMAIN>.md` ↔ `docs/plan/<DOMAIN>.md`) or **multi-layer feature pair** (`docs/features/architecture/<FEATURE>.md` ↔ `docs/features/plan/<FEATURE>.md`), then affected domain pairs when cross-layer (and ADR when required). Do not create `docs/plan/<FEATURE>.md` for multi-layer features.

**No code implementation** in Mode I unless the operator explicitly starts a **new** implement session afterward.

---

## When to use Mode I vs other audits

| Situation | Use |
|---------|-----|
| "Should we add X? Where does it belong? Does it already exist?" | **Mode I** — this orchestrator |
| "How mature is the MEMORY layer?" | Mode A — [`ORCHESTRATOR.md`](ORCHESTRATOR.md) |
| "Implement the next open plan row in RAG" | Mode B — [`IMPLEMENT_ORCHESTRATOR.md`](IMPLEMENT_ORCHESTRATOR.md) |
| "Close out the TOOLS domain layer" | Mode C — [`LAYER_COMPLETION_ORCHESTRATOR.md`](LAYER_COMPLETION_ORCHESTRATOR.md) |

---

## Before the session

**Recommended (simplest):** open a **new** Cursor agent chat and write in natural language, e.g.:

```text
Zrób audyt pomysłu: dodać integrację WhatsApp, żeby agent mógł zgłaszać użytkownikom postępy w zadaniach.
```

The project rule `.cursor/rules/intergrax-idea-audit.mdc` routes the agent to [`bootstrap/idea_audit.txt`](../bootstrap/idea_audit.txt).

**Also valid:**

- Reference the procedure: `na podstawie idea_audit przeprowadź audyt pomysłu …`
- Paste [`bootstrap/idea_audit.txt`](../bootstrap/idea_audit.txt) and add the idea **below** it in the same message.

**Do not** edit the bootstrap file — the idea always lives in the operator message.

**Skim once per session:** [`SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md).

---

## Live audit policy (no idea-results folder)

| Policy | Meaning |
|--------|---------|
| **Source of truth** | Live `docs/architecture/`, `docs/plan/`, and code |
| **Deliverable** | Operator-facing summary in chat |
| **Duplicate detection** | Search plan registers, architecture canon, code, and ADRs — not prior chat logs |
| **After operator approval** | Update affected **architecture + plan** domain pair(s); add ADR when significant |
| **No persistence layer** | Do not create `ideas_results/`, `IDEA_AUDIT.md`, or similar sidecar files |

Re-running the same idea months later is expected — each session re-reads current canon and code.

---

## Idea classification (agent infers from operator message)

| `IDEA_TYPE` | Primary domain pair(s) | Code search roots | Authoring guide |
|-------------|------------------------|-------------------|-----------------|
| `integration` | `INTEGRATIONS` | `intergrax/integrations/` | [`AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix K |
| `tool` | `TOOLS` | `intergrax/tools/` | [`TOOLS.md`](../architecture/TOOLS.md) · `intergrax/tools/USAGE.md` |
| `skill` | `SKILLS` | `intergrax/skills/` | [`SKILLS.md`](../architecture/SKILLS.md) |
| `agent` | `AGENT_CONTRACTS_AND_ASSEMBLY` | `agents/` | [`AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) |
| `application` | `TIER3_APPLICATION_ENVIRONMENT` | `applications/` | [`APPLICATION_CREATION_GUIDE.md`](../guides/APPLICATION_CREATION_GUIDE.md) |
| `harness_capability` | map via [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../guides/INTEGRAX_HARNESS_AUDIT_MAP.md) | `intergrax/` · `intergrax/runtime/` (Tier-1 only when justified) | hub [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md) |
| `improvement` | affected domain(s) from task routing | existing component paths | — |
| `analysis` | as inferred | as needed | — |
| `cross_cutting` | 2+ domain pairs | union of roots | safety fuse mandatory |

Task routing table: [`AGENTS.md`](../../AGENTS.md).

---

## Mandatory work cycle

### Step 0 — Duplicate / prior-art scan (live sources only)

Search **current** repository state:

1. `docs/plan/<DOMAIN>.md` — open rows, maintenance queues, similar IDs or slugs
2. `docs/architecture/<DOMAIN>.md` — existing contracts, catalog slugs, capabilities
3. Code paths from classification table — modules, manifests, registry entries, tests
4. `docs/plan/PLATFORM_FOUNDATION.md` §6.3 — product backlog
5. `docs/adr/` and tier-2/3 `adr/` — precedent decisions

Report overlaps in chat. **Do not** rely on prior Cursor sessions or sidecar audit files.

### Step 1 — Parse and classify

1. Restate the idea in one sentence (operator language).
2. Set `idea_type`, `primary_tier` (0–3), and `domain_pairs` (basenames).
3. List explicit **success criteria** and **non-goals** from the operator description.
4. Flag **Phase K / §6.3** product backlog touch — requires explicit reprioritization to implement.

### Step 2 — Read canon (narrow scope)

Read **only** what the classification requires:

1. `docs/intergrax_runtime_architecture.md` — hub / pair picker
2. `docs/guides/SYSTEM_INVARIANTS.md` — skim; P0 on violation
3. `docs/architecture/<DOMAIN>.md` + `docs/plan/<DOMAIN>.md` for each mapped pair
4. Relevant authoring guide (agent / application / extension) when Tier-2/3
5. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — sections for mapped audit-map layers only

Do **not** load all 22 domain pairs unless `cross_cutting` truly requires it.

### Step 3 — Inspect implementation (evidence required)

| Check | What to find |
|-------|----------------|
| **Already exists** | module, registry entry, scaffold output, test, manifest |
| **Partial overlap** | adjacent capability to extend vs duplicate |
| **In plan** | plan row ID, status, acceptance criteria |
| **Invariant risk** | `SYS-INV-*`, tier boundary, Nexus fork, vendor SDK in Tier-2 |
| **ADR precedent** | existing ADR in correct domain |

Evidence format: `` `path:symbol` ``, test name, or gate output — **no doc-only surveys**.

### Step 4 — Verdict matrix

| Verdict | Meaning |
|---------|---------|
| `already_implemented` | Done in code — point to artifacts |
| `in_plan` | Registered — cite row IDs |
| `partial_overlap` | Extend existing capability |
| `novel` | Propose architecture + plan rows |
| `blocked` | Invariant or safety-fuse conflict |
| `needs_clarification` | Ask concrete questions |
| `deferred_product` | Phase K / §6.3 without reprioritization |

### Step 5 — Architecture proposal (when `novel` or `partial_overlap`)

Specify in chat: tier, domain pair(s), contracts, reuse inventory, ADR need, scaffold commands, **draft plan row IDs** (English), tests/gates, suggested next implementation step.

### Step 6 — Operator checkpoint (Polish) — **STOP**

Deliver summary per bootstrap output structure. **Do not** edit architecture or plan yet.

Ask:

- Agreement with verdict?
- Approve proposed architecture outline?
- Approve plan row drafts?
- Apply doc updates now (architecture + plan)?
- Start implementation in a new session?

### Step 7 — Apply canon updates (only after explicit operator approval)

When the operator approves:

1. **Domain-layer work:** update `docs/architecture/<DOMAIN>.md` when contracts or capability surface changes; add or update rows in `docs/plan/<DOMAIN>.md`.
2. **Multi-layer feature work:** create or update `docs/features/architecture/<FEATURE>.md` and `docs/features/plan/<FEATURE>.md`, then update affected domain pairs. Domain-specific implementation rows still belong in owning `docs/plan/<DOMAIN>.md` files.
3. Create ADR in the correct domain when significant (`docs/adr/` or agent/app `adr/`)
4. Record **"no ADR needed"** with rationale when applicable
5. Add implementation-journal entry when the approved idea changes harness architecture or registers significant plan rows (see [`implementation-journal/README.md`](../implementation-journal/README.md)); otherwise record **"no journal needed"** in chat.

**Do not** commit unless the operator requests.

---

## Example operator message

```text
Zrób audyt pomysłu: dodać integrację WhatsApp do powiadomień agentów o postępie zadań.
Kryterium sukcesu: wysyłka przez ToolRuntime pod polityką Nexus.
Poza zakresem: inbound webhook, szablony Phase K.
```

Agent should classify → `INTEGRATIONS`, scan `intergrax/integrations/` for `twilio`, `notify.send`, deliver verdict in Polish, then STOP.

---

## Anti-patterns (forbidden)

- Writing idea-audit sidecar files (`IDEA_AUDIT.md`, `ideas_results/`, etc.)
- Shallow opinion without code/plan evidence
- Creating new Tier-0 mechanisms when an existing one fits
- Editing architecture/plan before operator approval
- Agent-specific Nexus forks for one business agent
- Starting Phase K / §6.3 without explicit reprioritization

---

## Related workflows after approval

| Next step | Bootstrap / guide |
|-----------|-------------------|
| Implement harness plan item | `bootstrap/04_implement_plan_one_domain.txt` |
| Create agent | [`AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) |
| Create application | [`APPLICATION_CREATION_GUIDE.md`](../guides/APPLICATION_CREATION_GUIDE.md) |
| Extension / plugin | [`EXTENSION_AUTHOR_GUIDE.md`](../guides/EXTENSION_AUTHOR_GUIDE.md) |
