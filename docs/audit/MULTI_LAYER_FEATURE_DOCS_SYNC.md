<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Multi-layer Features — Project Documentation Visibility Sync Instruction

**Status:** Docs-sync control prompt (copy-paste for Cursor / LLM agent)  
**Target branch:** `development`  
**Mode:** documentation visibility sync only  
**Runtime implementation:** forbidden in this task  
**Primary goal:** update all high-level project description and navigation documents so the existence of multi-layer feature documentation is visible, linked, and correctly explained.

---

## Why this instruction exists

Intergrax now has two paired documentation structures:

```text
docs/architecture/<DOMAIN>.md           ↔ docs/plan/<DOMAIN>.md
docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
```

The second structure is important because some capabilities are not single architecture layers. They cut across multiple domains and need their own coherent architecture/plan pair while still preserving domain ownership.

This fact must be visible in project-level documentation, not hidden only in `docs/features/README.md`.

---

## How to use

1. Open Cursor on repository `jakbuczarnecki/intergrax`.
2. Checkout branch `development`.
3. Start a fresh agent chat.
4. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
5. Do not add runtime implementation goals.
6. Let Cursor produce one focused documentation commit.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

repo: jakbuczarnecki/intergrax
branch: development
mode: multi-layer-feature-docs-visibility-sync
runtime_code_changes: forbidden
commit_policy: one focused docs commit

# ═══ END USER CONFIG ═══

# TASK: Update project documentation to expose multi-layer feature docs

You are working on the Intergrax Harness AI repository.

Your task is to update all high-level project description, navigation, audit, and onboarding documents so they clearly explain that Intergrax has **two paired documentation structures**:

```text
docs/architecture/<DOMAIN>.md           ↔ docs/plan/<DOMAIN>.md
docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
```

This is a documentation visibility sync only.

Do **not** implement runtime code.
Do **not** create Python modules.
Do **not** modify feature runtime plans beyond references/navigation.
Do **not** do a repo-wide rewrite.
Do **not** create `docs/plan/<FEATURE>.md` for multi-layer features.

---

## 0. Mandatory working rules

Before editing, state:

```text
read scope:
edit scope:
tests/checks:
```

Rules:

- Read `docs/features/README.md` first.
- Read `docs/intergrax_runtime_architecture.md`.
- Read `docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`.
- Read only relevant sections of other high-level docs.
- Use grep/path filters before opening large files.
- Do not load full implementation plans unless needed for navigation context.
- Do not load `docs/audit_results/`.
- Do not use subagents.
- Make minimal, precise edits.
- Preserve existing domain architecture/plan 1:1 rule.
- Preserve new feature architecture/plan 1:1 rule.
- Make one focused commit at the end.

---

## 1. Core concept that must be added everywhere relevant

Add a concise explanation of this model wherever project documentation describes architecture, plans, roadmap, audit workflow, or documentation topology:

```text
Intergrax uses two paired documentation structures:

1. Domain-layer pairs:
   docs/architecture/<DOMAIN>.md ↔ docs/plan/<DOMAIN>.md
   These describe individual architecture layers and their implementation plans.

2. Multi-layer feature pairs:
   docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
   These describe cross-layer capabilities that coordinate changes across multiple domains.

Feature docs coordinate cross-domain capability delivery. Domain docs remain the source of truth for domain-owned architecture and implementation rows.
```

Do not over-explain. Add short paragraphs, navigation rows, or doc-map entries.

---

## 2. Mandatory files to inspect and update if relevant

Inspect these files and update them where the concept naturally belongs.

### 2.1 Public/project entry points

```text
README.md
EVALUATION_GUIDE.md
FAQ.md
USE_CASES.md
ROADMAP.md
```

Minimum required update:

- `README.md` must mention `docs/features/README.md` and the paired feature architecture/plan structure.

### 2.2 Documentation navigation and architecture hubs

```text
docs/intergrax_runtime_architecture.md
docs/DOCUMENTATION_MAP.md
docs/features/README.md
```

If `docs/intergrax_runtime_architecture.md` already contains the new structure, only verify and refine links if needed.

### 2.3 Strategy and authoring guides

```text
docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md
docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md
docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md
docs/guides/LAYER_COMPLETION_MODE.md
docs/guides/AGENT_INSTRUCTIONS.md
```

Update only if the document describes documentation structure, architecture/plan workflow, audit scope, or project navigation.

### 2.4 Audit and bootstrap docs

```text
docs/audit/README.md
docs/audit/IDEA_AUDIT_ORCHESTRATOR.md
docs/audit/ORCHESTRATOR.md
docs/audit/IMPLEMENT_ORCHESTRATOR.md
docs/audit/LAYER_COMPLETION_ORCHESTRATOR.md
docs/bootstrap/README.md
docs/bootstrap/idea_audit.txt
```

Update these so future idea audits know that cross-layer features should be documented under `docs/features/architecture` and `docs/features/plan`, not as plan-only files under `docs/plan`.

### 2.5 Machine-readable / AI-navigation docs

Inspect and update if present:

```text
llms.txt
AGENTS.md
.cursor/rules/*.mdc
```

Important: if `.cursor/rules` contains rules about documentation, architecture, plan, or token budget, add the feature-docs rule there too.

---

## 3. Required README.md update

`README.md` must include the feature documentation concept in at least one visible place.

Recommended locations:

1. `Current platform maturity` paragraph, after the sentence about `docs/architecture/<DOMAIN>.md` ↔ `docs/plan/<DOMAIN>.md`.
2. `Start here` table, with a row such as:

```markdown
| Exploring multi-layer platform features | [Multi-layer feature docs](docs/features/README.md) |
```

3. Documentation index section if present.

Do not rewrite README. Add minimal precise text.

---

## 4. Required wording constraints

Use consistent terminology:

- **domain-layer pair**
- **multi-layer feature pair**
- **feature architecture**
- **feature plan**
- **cross-layer capability**
- **domain ownership remains authoritative**

Avoid ambiguous wording:

- Do not call feature plans “domain plans”.
- Do not say feature docs replace domain docs.
- Do not say every feature must become a layer.
- Do not create `docs/plan/<FEATURE>.md`.

---

## 5. Required final state

After this sync, a new reader should be able to discover the feature docs from:

- `README.md`,
- `docs/intergrax_runtime_architecture.md`,
- `docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`,
- `docs/audit/README.md`,
- `docs/DOCUMENTATION_MAP.md` if present,
- relevant bootstrap/audit guidance,
- AI navigation docs if present.

The final state must preserve:

```text
docs/architecture/<DOMAIN>.md           ↔ docs/plan/<DOMAIN>.md
docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
```

---

## 6. Checks

Run lightweight doc checks only.

Required if available:

```bash
uv run python scripts/audit/check_docs_domain_pairs.py
```

Also run any obvious lightweight documentation/navigation checks discovered by:

```bash
ls scripts | grep -E "check.*doc|check.*arch|check.*plan|documentation|llms"
```

Do not run the full test suite.

If checks cannot be run, state why.

---

## 7. Expected final response

Return:

```text
Outcome:
- what changed

Files updated:
- ...

Feature visibility added to:
- README.md
- docs/intergrax_runtime_architecture.md
- ...

Checks:
- commands run / not run and why

Commit:
- <sha> <message>

Next step:
- run feature-specific docs sync for TOKEN_OPTIMIZATION domain rows, or start first implementation slice if docs are already synced.
```

---

## 8. Commit requirement

Create one focused commit on `development`.

Suggested commit message:

```text
docs: expose multi-layer feature documentation structure
```

Do not group runtime implementation with this commit.

---END PROMPT---
