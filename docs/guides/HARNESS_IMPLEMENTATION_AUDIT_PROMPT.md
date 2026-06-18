# Harness Implementation Audit — Copy-Paste Prompt

**Purpose:** repeatable LLM prompt for Intergrax Harness AI implementation audits.  
**Procedure source:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Per-domain deep audits:** [`audit/README.md`](audit/README.md) — copy-paste prompts for each of the 22 domain pairs (RAG, Tools, Memory, Context Engineering, …)

---

## How to use

### Single domain (recommended for engine-depth audits)

For one domain pair (e.g. RAG, Tools, Memory), use the dedicated prompt — **do not** rewrite ad-hoc instructions:

1. Open [`audit/<DOMAIN>.md`](audit/README.md#domain-index-22-pairs) (e.g. [`audit/RAG.md`](audit/RAG.md)).
2. Copy `---BEGIN PROMPT---` … `---END PROMPT---` into a new agent chat.
3. Set `mode` in USER CONFIG.

Domain prompts include shared observability/security/scale checklists plus domain-specific dimensions.

### Multi-layer or full-platform audit

For **all 22 domain pairs** in one session, use [`audit/bootstrap/01_audit_all_domains.txt`](audit/bootstrap/01_audit_all_domains.txt) (Mode A) or [`audit/README.md`](audit/README.md) for other modes.

### What to copy

Copy the block from `---BEGIN PROMPT---` through `---END PROMPT---` (markers optional).

**After pasting, edit only the USER CONFIG block at the very top** — 4 lines (`scope`, `layer`, `phase`, `mode`). Everything below stays unchanged.

You do **not** need to scroll or search inside §4 anymore.

### USER CONFIG fields

| Field | When | Values |
|-------|------|--------|
| `scope` | always | `A` = one layer · `B` = one phase · `C` = all 32 layers |
| `layer` | scope `A` | e.g. `Policy and Governance` |
| `phase` | scope `B` | `Phase 1` · `Phase 2` · `Phase 3` |
| `mode` | always | `audit-only` · `audit-and-fix` |

Leave `layer` or `phase` blank when not used.

### Presets (uncomment one block in USER CONFIG)

**Single layer, report only:** `scope: A` · `layer: Policy and Governance` · `mode: audit-only`

**Phase 2, report only:** `scope: B` · `phase: Phase 2` · `mode: audit-only`

**Full architecture, update plan:** `scope: C` · `mode: audit-and-fix`

### Where to paste

Cursor Agent (or any LLM with repo access) — first message in a new chat.

### After the run

Model must end with **Completion Summary** (§8).

---

---BEGIN PROMPT---

# ═══ USER CONFIG — edit only this block ═══

scope: C
layer: 
phase:
mode: audit-and-fix

# scope: A | B | C
# layer: required when scope=A — see §4.1 catalog
# phase: required when scope=B — Phase 1 | Phase 2 | Phase 3
# mode: audit-only | audit-and-fix

# ═══ END USER CONFIG — do not edit below on reuse ═══

# TASK: Intergrax Harness Implementation Audit

You are an **implementation audit agent** for the Intergrax Harness AI platform.

**First:** read **USER CONFIG** above and set audit scope and mode from it. Do not ask the user to confirm unless a required field is missing or invalid.

Compare target architecture, current architecture, implementation plan, source code, and tests. Produce evidence-backed findings and maturity scores.

**Do not implement code in this task** unless the user explicitly requests a separate implementation pass.

---

## 1. Mission

Intergrax is a **Harness AI / Agent OS** platform. The durable product is the **Harness** (runtime, governance, registries, observability). **Agents** are replaceable execution units.

**Strategic relationship:**

```text
Harness → Runtime → Agents → Applications → Products
```

**Audit goal:** Identify concrete gaps between target and reality, score layer maturity, and produce actionable evidence — without shallow “everything is done” declarations.

**Success criterion:** A layer is complete only when architecture, implementation plan, code, tests, and documentation **all** provide verifiable evidence.

---

## 2. Canonical References (read in this order)

| # | Document | Role | What to extract |
|---|----------|------|-----------------|
| 1 | `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | **Target state** | Ideal Harness AI: 9 logical layers, design principles, domain entities, reference flow, extension points |
| 2 | `docs/intergrax_runtime_architecture.md` | **Current canonical architecture** | Four-tier model (Tier-0→3), §42 Unified Execution Runtime, forbidden patterns, implementation rules |
| 3 | `docs/intergrax_runtime_architecture.md` | **Implementation roadmap & status** | Phase trackers (Q, Q+, R, S, U, V, W-OPS, H-APP, MEM, DX, AA…), gate counts, Done/In-progress rows |
| 4 | `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | **Audit procedure** | Layer-specific audit questions, typical gaps, scoring model, output format, global rules |
| 5 | `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix H** | **Governance control plane (authoring)** | `ApplicationEnvironmentProfile` map, `RuntimePolicyBundle`, security profile, observability mandatory vs optional, verification commands — use when auditing §5 Policy and §21 Observability |
| 6 | `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix I** | **Orchestration control plane (authoring)** | Nexus runners, `ExecutionGraph`, `DelegationSpec`, hooks, planning strategies, customization surfaces — use when auditing §7–§10; implementation closeout: plan **Phase ORCH** (**Done**) |
| 7 | `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix J** | **Tools & skills control plane (authoring)** | `ToolProfile`, `SkillProfile`, `catalog_runtime_bridge`, `SkillResolverProtocol`, `ToolRuntime`, conformance checks — use when auditing §11–§12; implementation closeout: plan **Phase TS** (**Done**) |
| 8 | `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix K** | **Integration & RAG control plane (authoring)** | `IntegrationProfile`, `integration_runtime_bridge`, `rag_runtime_bridge`, health probes, `RetrievalService` — use when auditing §13–§14; canon: [`architecture/RAG.md`](architecture/RAG.md); closeout: **Phase INT** + **Phase RAG** (**Done**) |
| 9 | `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix L** | **Context engineering control plane (authoring)** | `ContextProfile`, `context_runtime_bridge`, `context_wiring`, `ContextManager`, `ContextBudgetPolicy` — use when auditing §16; closeout: **Phase CTX** (**Done**) |
| 10 | `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix M** | **Prompt registry control plane (authoring)** | `PromptProfile`, `prompt_runtime_bridge`, `prompt_wiring`, `YamlPromptRegistry`, `PromptRegistryProtocol` — use when auditing §17; closeout: **Phase PE** (**Done**) |

**Always distinguish these eleven views — never conflate them:**

- **Target** → `IDEAL_HARNESS_AI_ARCHITECTURE.md`
- **Current architecture** → `intergrax_runtime_architecture.md`
- **Plan** → `intergrax_runtime_architecture.md`
- **Governance authoring** → `guides/AGENT_CREATION_GUIDE.md` Appendix H
- **Orchestration authoring** → `guides/AGENT_CREATION_GUIDE.md` Appendix I
- **Tools/skills authoring** → `guides/AGENT_CREATION_GUIDE.md` Appendix J
- **Integration/RAG authoring** → `guides/AGENT_CREATION_GUIDE.md` Appendix K
- **Context engineering authoring** → `guides/AGENT_CREATION_GUIDE.md` Appendix L
- **Prompt registry authoring** → `guides/AGENT_CREATION_GUIDE.md` Appendix M
- **Implementation** → source code under `intergrax/`, `agents/`, `applications/`
- **Verification** → tests, CI gates, scripts

---

## 3. Ideal Harness AI — Compact North Star

**Design principles (policy-first Harness):**

1. Policy-first — nothing executes without policy/permission/constraint checks
2. Composable-by-default — small, isolated, replaceable components
3. Trace-everything — every decision and invocation is traceable
4. Safe-failure — failures anticipated, classified, handled
5. Deterministic-enough — reproducibility where practical
6. Human-governed autonomy — HITL paths for high-risk actions
7. Progressive extensibility — providers, tools, skills, protocols via registries

**9 logical layers (ideal):** Interface → Identity & Trust → Policy & Governance → Orchestration → Cognition (LLM + modality planes A/B/C) → Capability (Tools + Skills + Integrations) → Memory & Knowledge → Reliability & Runtime → Observability & Operations

**Core domain entities:** `TaskEnvelope`, `Run`, `Step`, `DecisionRecord`, `ToolInvocation`, `PolicyDecision`, `MemoryArtifact`, `Observation`, `Incident` — each with `trace_id`, `run_id`, `tenant_id`, `version`, `created_at`.

**Intergrax four-tier mapping:**

```text
Tier-0 Platform     — universal mechanisms (LLM, RAG, memory, policy, registries)
Tier-1 Nexus        — domain-agnostic Agent OS / runtime (§42)
Tier-2 Agents       — composable workers (no vendor SDKs, no direct integrations)
Tier-3 Applications — domain packages composing runtime + agents + tools + policies
```

**Critical rule:** Reuse existing Tier-0 mechanisms. Do **not** create duplicate universal components.

---

## 4. Audit Scope (from USER CONFIG)

Interpret **USER CONFIG** at the top of this prompt:

| `scope` | Action |
|---------|--------|
| `A` | Audit exactly the layer named in `layer` (§4.1 catalog). Highest precision. |
| `B` | Audit all thematic layers for `phase` per `INTEGRAX_HARNESS_AUDIT_MAP.md` §7 (see §4.2). |
| `C` | Audit all 32 layers in §4.1. Full-architecture rules in §4.3. |

| `mode` | Action |
|--------|--------|
| `audit-only` | Report only. **Do not edit any files.** |
| `audit-and-fix` | Report, then update `intergrax_runtime_architecture.md` (and architecture doc if drift found). **No code or test changes.** See §5 Step 8. |

If `scope: A` and `layer` is empty → stop and ask for layer name.  
If `scope: B` and `phase` is empty → stop and ask for phase.  
If `scope: C` → ignore `layer` and `phase`.

### §4.2 Phase layer lists (`INTEGRAX_HARNESS_AUDIT_MAP.md` §7)

- **Phase 1 — Core Harness Integrity (10):** Strategic Harness Model; Tier Model and Dependency Boundaries; Execution Runtime and Agent OS; Policy and Governance; Tool / Skill / Integration Separation; Context Engineering; Observability and Telemetry; Error Handling and Reliability; Evaluation and Benchmarking; Architecture Governance and Documentation Loop
- **Phase 2 — Capability Platform (8):** LLM and Model Adapter Layer; Prompt Engineering and Prompt Registry; RAG and Retrieval Layer; Memory Layer; Registry Architecture; Capability Graph Architecture; Modality, Vision, Audio and Dedicated ML; Developer Experience, Scaffold and Lab
- **Phase 3 — Production Readiness (8):** Identity, Trust and Tenancy; Security and Data Governance; Cost and Resource Governance; Product Environment and Tier-3 Applications; Testing, CI and Architecture Gates; Agent Lifecycle Governance; Operational Excellence and SLOs; Incident Management and Runbooks

For **Tool / Skill / Integration Separation** (Phase 1): audit Tool Layer, Skill Layer, and Integration Layer together (catalog layers 11–13).

For phase scope: run §5 per layer; one consolidated report with per-layer subsections.

### §4.3 Full architecture rules (`scope: C`)

1. Process layers sequentially by phase (Phase 1 → 2 → 3).
2. Per layer: audit questions, L0–L4 score, gaps — do not skip layers.
3. Output a **layer scorecard** for all 32 layers.
4. Detailed gap analysis for **Critical** and **High** only; Medium/Low in scorecard.
5. Never declare platform complete without evidence for every layer.
6. Inspect code/tests per layer — no documentation-only survey.

### §4.1 Full layer catalog

**Phase 1 — Core Harness Integrity**

1. Strategic Harness Model
2. Tier Model and Dependency Boundaries
3. Interface and Task Intake
4. Identity, Trust and Tenancy
5. Policy and Governance
6. LLM and Model Adapter Layer
7. Reasoning, Planning and Cognition
8. Execution Runtime and Agent OS
9. Orchestration, Scheduler and Execution Graph
10. Subagents and Multi-Agent Coordination
11. Tool Layer
12. Skill Layer
13. Integration Layer
14. RAG and Retrieval Layer
15. Memory Layer
16. Context Engineering Layer
17. Prompt Engineering and Prompt Registry
18. Agent Assembly and Agent Contracts
19. Registry Architecture
20. Capability Graph Architecture
21. Observability and Telemetry
22. Error Handling and Reliability
23. Security and Data Governance
24. Cost and Resource Governance
25. Evaluation and Benchmarking
26. Testing, CI and Architecture Gates
27. Developer Experience, Scaffold and Lab
28. Product Environment and Tier-3 Applications
29. Modality, Vision, Audio and Dedicated ML
30. Operational Excellence and SLOs
31. Agent Lifecycle Governance
32. Architecture Governance and Documentation Loop

For each audited layer: read **Purpose**, **Audit Questions**, and **Typical Gaps** in `INTEGRAX_HARNESS_AUDIT_MAP.md` §8.

---

## 5. Mandatory Audit Workflow

Execute these steps **in order** for each layer in scope. Do not skip steps.

### Step 1 — Read references

1. Read the relevant sections of the canonical documents for this layer (IDEAL, architecture canon, implementation plan, audit map; for **§5 Policy** and **§21 Observability** read Appendix H; for **§7–§10** planning/graph/subagents read Appendix I).
2. Note which `intergrax_runtime_architecture.md` phases/appendices claim Done vs In-progress for this area.

### Step 2 — Map to codebase

Identify and inspect **concrete** files/modules/tests for this layer. Examples:

- Runtime: `intergrax/runtime/nexus/`, `intergrax/harness/`
- Policy: policy engine, `ToolRuntime`, `RuntimePolicyBundle`
- Capabilities: `intergrax/tools/`, `intergrax/skills/`, `intergrax/integrations/`
- Agents: `agents/`, `intergrax/agents/`
- Applications: `applications/`
- Tests: `tests/unit/`, `tests/integration/`, `tests/acceptance/`

Use search/grep — do not rely on memory or assumptions.

### Step 3 — Answer layer audit questions

For each audit question in `INTEGRAX_HARNESS_AUDIT_MAP.md` for this layer, provide:

- **Answer:** Yes / Partial / No / Unknown
- **Evidence:** file path + symbol/test name (or explicit “not found” after search)

### Step 4 — Gap analysis

List **concrete** gaps between:

- Target (`IDEAL_HARNESS_AI_ARCHITECTURE.md`)
- Current arch (`intergrax_runtime_architecture.md`)
- Plan status (`intergrax_runtime_architecture.md`)
- Actual code/tests

Each gap must be specific, e.g. “`NexusLoop` contains agent-specific branch for legal_agent” — not “policy could be better”.

### Step 5 — Risk assessment

Classify each gap: **Critical / High / Medium / Low** with impact explanation.

### Step 6 — Maturity scoring

Score each layer independently:

```text
L0 — Fragmented: local only, no governance/telemetry
L1 — Operational MVP: basic mechanism, weak tests
L2 — Scalable Harness: modular, registered, reusable across agents
L3 — Production Harness OS: full policy, telemetry, tests, docs, SLOs
L4 — Adaptive Agent OS: closed feedback loops, evaluation-driven improvement
```

Report: **Score before**, **Target score for current milestone**.  
**Score after** applies only when a prior remediation iteration actually changed implementation (not in `audit-and-fix` doc-only mode).

### Step 7 — Verification (mandatory)

Every claim must be backed by evidence. Run applicable checks:

```bash
# Primary regression gate
uv run pytest -m gate -q

# Layer-relevant targeted tests (adapt to audited layer)
uv run pytest tests/unit/<relevant>/ -q
uv run pytest tests/integration/<relevant>/ -q
uv run pytest tests/acceptance/<relevant>/ -q

# Architecture boundary scripts (when relevant)
python scripts/check_harness_no_getattr.py
python scripts/check_agents_vendor_imports.py
python scripts/check_integration_vendor_imports.py
python scripts/check_scaffold_harness_alignment.py
```

If a command cannot run, state why and what manual evidence substitutes it.

### Step 8 — Documentation remediation (only if Mode = audit-and-fix)

Update documentation to close the **plan ↔ reality** gap:

1. **`intergrax_runtime_architecture.md`** (primary):
   - Fix incorrect status markers (Done/In-progress) backed by audit evidence
   - Add new task rows with phase IDs, priority, owner area, acceptance criteria
   - Update appendix traceability matrices (C, D, E, G, …) when audit source is known
   - Add or refresh audit traceability row in §0 doc model table if applicable
2. **`intergrax_runtime_architecture.md`** (secondary, only if drift found):
   - Add missing contract references, tier rules, or §42 alignment notes
   - Do not rewrite unrelated sections
3. **Do not** modify source code or tests in this mode.

After doc updates, list every file changed and summarize what was corrected vs added.

### Step 9 — Out-of-scope findings

Issues outside the current layer/phase → record as:

```md
Out-of-scope finding:
- Area:
- Risk:
- Suggested next audit layer:
```

Do **not** address out-of-scope items in `audit-and-fix` unless they are plan traceability corrections.

---

## 6. Forbidden Patterns

- “The entire architecture is complete” / “All issues resolved” / “Platform fully aligned” — unless all in-scope layers were audited with evidence and no Critical gaps remain undocumented
- Marking a layer Done without file/test/gate evidence
- Confusing plan status (Done in markdown) with actual code reality
- Implementing code or tests during an audit task (unless user explicitly requests implementation separately)
- Creating duplicate Tier-0 mechanisms (even in recommendations — always prefer wiring existing modules)
- Claims without code citations or test results
- Shallow full-architecture surveys that skip per-layer code inspection

**Allowed completion wording:**

```text
Scope audited: <layer | phase | full architecture>
Layers covered: <list>
Score summary: <per-layer L0-L4 table>
Critical/High gaps: <count + top items>
Plan updated: yes/no
Remaining risks: ...
Next recommended audit: ...
```

---

## 7. Required Output Format

### Single-layer or single-phase scope

```md
# Audit Result: <Layer Name or Phase Name>

## 1. Scope
What was audited (files, modules, tests, doc sections).

## 2. Target State
What `IDEAL_HARNESS_AI_ARCHITECTURE.md` requires.

## 3. Current State
What Intergrax implements today (architecture doc + code + tests).

## 4. Gap List
| # | Gap | Target ref | Current evidence | Severity |
|---|-----|------------|------------------|----------|

## 5. Risk Assessment
Impact of gaps (Critical/High/Medium/Low).

## 6. Required Architecture Updates
Changes needed in `intergrax_runtime_architecture.md`.

## 7. Required Implementation Plan Updates
Changes needed in `intergrax_runtime_architecture.md` (phase IDs, appendix rows).
(In `audit-and-fix` mode: mark which of these were applied.)

## 8. Recommended Code Changes
Concrete future implementation work (file-level). **Not executed in this audit.**

## 9. Recommended Tests
Unit / integration / contract / acceptance / architecture-boundary tests. **Not executed in this audit.**

## 10. Definition of Done
Precise criteria to mark this layer complete.

## 11. Evidence
- Files inspected:
- Files changed (if audit-and-fix):
- Tests/gates run + results:
- Documentation updated:

## 12. Remaining Risks

## 13. Next Recommended Audit

## Maturity Score
- Score before: L?
- Target score: L?
- Evidence supporting score:
```

### Full-architecture scope (add after per-layer work)

```md
# Full Architecture Audit — Layer Scorecard

| # | Layer | Score | Critical gaps | High gaps | Plan status accurate? |
|---|-------|-------|---------------|-----------|------------------------|

# Top Critical & High Gaps (detailed)

# Cross-Layer Themes
Patterns spanning multiple layers (e.g. policy bypass, tier violations, missing telemetry).

# Recommended Plan Restructure
New phases, reprioritized tasks, or appendix updates for `intergrax_runtime_architecture.md`.

# Recommended Next Focused Audits
Which single layers deserve deep follow-up.
```

---

## 8. Completion Summary (mandatory closing block)

```md
# Completion Summary

Scope audited: <single layer | phase | full architecture>
Layers covered: <list or count>
Mode: <audit-only | audit-and-fix>
Score summary: <L0-L4 per layer or aggregate>

## Evidence
- Files inspected: <count + key paths>
- Documentation updated: <yes/no — list files if audit-and-fix>
- Implementation plan updated: <yes/no — summarize changes>
- Tests/gates run: <commands + results>

## Remaining Risks

## Out-of-Scope Findings

## Next Recommended Audit
```

---

## 9. Final Rule

Intergrax is evaluated by **Harness layer maturity** — not by feature count.

A layer is complete only when: correctly designed, correctly implemented, policy-governed, observable, testable, documented, reusable, and **verifiable with evidence**.

An audit **fixes the plan**; implementation is a separate, explicitly requested step.

Begin the audit now.

---END PROMPT---
