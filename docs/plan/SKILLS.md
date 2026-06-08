# Skills — Implementation Plan

**Architecture (1:1):** [`architecture/SKILLS.md`](../architecture/SKILLS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

### 6.1c Harness implementation queue — tools/skills closeout (closed)

**Purpose:** Single ordered list for **Phase TS** (Band 2k). **Closed 2026-06-02** — all TS rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **TS-DOC.1–2** | Docs | **Done** | Appendix J + cross-refs | Author map complete |
| 2 | **TS-1** | Code | **Done** | `catalog_runtime_bridge` + `RuntimeConfig.skill_profile` | `test_catalog_runtime_bridge.py` |
| 3 | **TS-2** | Code | **Done** | Harness host `resolve_llm_adapter` wiring | `test_harness_host_runtime_llm.py` |
| 4 | **TS-3** | Code | **Done** | `SkillResolverProtocol` | skill resolver tests green |

**Suggested PR order (complete):** TS-1 → TS-2 → TS-3 → TS-DOC.*.

**Explicitly excluded:** K.1, K.2, new product tools/skills, business agent packs — [§6.3a](#63a-business-backlog-register-consolidated).### 6.1aa Harness implementation queue — memory platform (closed)

**Purpose:** Phase MEM execution queue — **closed 2026-06-02** (48/48 Done). Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **MEM-1.1–MEM-1.4** | Code | **Done** | H-APP `MemoryProfile` + `ContextProfile.budget` + SQLite session → `RuntimeConfig` | MEM-1.5 gate test green |
| 2 | **MEM-2.1–MEM-2.3** | Code | **Done** | `SQLiteUserProfileStore` + bundle wiring + unit tests | LTM survives restart on sqlite profile |
| 3 | **MEM-1.6** | Docs/status | **Done** | H-APP.4.3 → **Done** | Bridge complete |
| 4 | **MEM-4.1–MEM-4.3** | Test | **Done** | Session + LTM + full-stack memory gates | acceptance/integration green |
| 5 | **MEM-5.1–MEM-5.2** | Test/Docs | **Done** | `engine_history_layer` tests + compression docs | unit + guide |
| 6 | **MEM-3.1–MEM-3.3** | Code | **Done** | Memory store plugin EP + reference fixture | bootstrap + gate |
| 7 | **MEM-0.3–MEM-DOC.*** | Docs | **Done** | Author cookbooks + Appendix G sync | guide updated |
| 8 | **MEM-6.*–MEM-7.*** | Code | **Done** | Retention enforcement + memory hooks | P2 after P0/P1 |
| 9 | **MEM-8.*–MEM-9.*** | RFC | **Done (RFC)** | Product memory layer + entity graph design | §6.3 gate for implementation |

**Suggested PR order:** See [Phase MEM — Suggested PR order](#mem--paydown-log).

**Explicitly excluded:** K.1, K.2, Mem0 SaaS product, entity graph ship (RFC only), business agent memory.

---

## Phase TS — Tools & skills control plane closeout

**Status:** **Done** (2026-06-02) — **5/5** deliverables Done (TS-DOC.* + TS-1–3); gate **589 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §11–§12; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix J**.

**Priority ladder:** **Band 2k** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bc](#62bc-phase-ts-execution-order-band-2k--closed) · queue: [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed)

**Delivery rule:** One **TS-*** ID per PR → update master table + §6.1c + paydown log below → `pytest -m gate` + §6.1 scripts green.

### TS — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TS-DOC.1 | TS0 | **Appendix J** — tools & skills control plane map (§J.1–J.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| TS-DOC.2 | TS0 | **Cross-ref sync** — plan, README, AUDIT_MAP §11–§12, audit prompt ref #7 | **Done** | Medium | `docs/*` | Links resolve |
| TS-1 | TS1 | **`catalog_runtime_bridge.py`** — `tool_profile` / `skill_profile` on `RuntimeConfig` via `materialize_runtime_config` | **Done** | **Critical** | `catalog_runtime_bridge.py`, `runtime_config_bridge.py`, `config.py` | `test_catalog_runtime_bridge.py` |
| TS-2 | TS2 | **Harness host LLM wiring** — `resolve_llm_adapter(env)` → `build_nexus_loop_from_environment` | **Done** | High | `harness_host_runtime.py` | `test_harness_host_runtime_llm.py` |
| TS-3 | TS3 | **`SkillResolverProtocol`** — typed contract for skill composition resolution | **Done** | Medium | `skills/resolver.py`, `contract_resolution.py` | existing skill resolver tests green |

**Residual (not TS scope — track separately):** legacy `use_rag`/`use_websearch` booleans in `engine_planner` / `tool_gateway` (deprecation warnings; `check_legacy_tool_plan_booleans.py`).

### TS — Paydown log

| Date | TS ID | Summary |
|------|-------|---------|
| 2026-06-02 | TS-DOC.1, TS-DOC.2 | Appendix J + cross-refs; AUDIT_MAP §11–§12 authoring map |
| 2026-06-02 | TS-1, TS-2, TS-3 | Catalog runtime bridge, harness LLM wiring, SkillResolverProtocol; gate **589** |

**Phase TS complete when:** TS-1–3 + TS-DOC.* **Done**; §6.1c queue closed; Appendix J has no “planned wiring” gaps; gate **589** green. **Status: complete (2026-06-02).**

---

---

### Phase R — Harness AI Alignment (post-audit 2026-06-01)

**Source:** Harness AI philosophy audit (scaffold, harness, LLM, tool vs skill, context engineering, subagents, policy) — traceability in **Appendix E**.  
**Status:** **Done (MVP)** (2026-06-01). **Prerequisite met:** Phase **Q+ Done**.  
**Goal:** Intergrax vocabulary and Tier-0 modules align with industry harness terminology **without** breaking Integration → Tool → Agent stack; add **Skill Library** for reuse and external compatibility.  
**Principle:** evolve, not rewrite · skills **compose** tools (never replace `ToolRuntime`) · one R.* ID per PR · gate green.

**Out of scope for Phase R:**

- Nested full harness per child (Cursor 1:1 subagent OS) — use graph delegation first (R-Delegate)
- Auto-discovery of skills from filesystem without validation
- Mandatory migration of all Tier-2 agents to skills in one release

**Phase R (MVP) complete:** Appendix E 100% **Done** or **Won't fix**; §0 Phase R row **Done**; gate **450 passed** (2026-06-01). Further skill catalog expansion is product work, not a harness gate.

---

#### R.0 — Canon, ADR, terminology (do first)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R.0.1 | **ADR: Skill layer Option 2** — reject “skills = tools only”; document four-layer model | **Done** | **Critical** | Architecture §7.1.8, §5.3 | Option 1 listed as rejected with rationale |
| R.0.2 | **Canon sections** — §5.3 Harness mapping, §7.1.8 Skills, §28.1 Context engineering, §42.14.3 Delegation, §42.11.4 Policy bundle | **Done** | **Critical** | `intergrax_runtime_architecture.md` | Cross-linked from plan §0 |
| R.0.3 | **Remove tool/skill conflation** in code docstrings | **Done** | High | `tools/core/contracts.py` | `ToolContract` describes **tool** only |
| R.0.4 | **README navigation** — Phase R, skills layer in root + docs README | **Done** | Medium | `/README.md`, `docs/README.md` | GitHub landing + docs index mention skills |

**Delivery rule:** Same as §6.1 — one R.* ID → PR → update Appendix E status → gate.

---

#### R-Skill — Skill Library (Tier-0)

**Problem:** Integrations and tools are production-grade; **skills are not**. Agents duplicate prompts, tool allow-lists, and policy fragments. External harness ecosystems (Cursor skills, internal markdown packs) cannot plug in without a **validated manifest**.

**Target layout:**

```text
intergrax/skills/
├── core/                   # SkillContract, SkillManifest, SkillProvider protocol
├── registry/               # SkillCatalog, SkillProfile, register_default_skills()
├── importers/              # cursor_skill_md.py, … (validate → SkillManifest)
├── _shared/
└── providers/
    └── <domain>/           # e.g. legal/, research/
        ├── manifest.py     # SkillManifest instance(s)
        ├── prompts.yaml    # or Prompt Registry refs
        └── USAGE.md
```

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Skill.1 | **`SkillManifest` + `SkillContract`** — frozen manifest: `skill_id`, `version`, `description`, `tool_ids`, `prompt_instruction_ids`, `policy_fragment_id`, `risk_tier`, `tags` | **Done** | **Critical** | `intergrax/skills/core/contracts.py` | Pydantic/jsonschema round-trip test |
| R-Skill.2 | **`SkillRegistry` + `SkillProfile` + `SkillCatalog`** — mirror Tool registry pattern | **Done** | **Critical** | `intergrax/skills/registry/` | `build_registry_from_profile()` |
| R-Skill.3 | **`SkillResolver`** — given `skill_ids`, produce resolved `allowed_tools` ∪, prompt pack refs, policy fragments; **no LLM execution** in resolver | **Done** | **Critical** | `intergrax/skills/resolver.py` | Unit: two skills merge tool lists with conflict rules |
| R-Skill.4 | **Tier-3 wiring** — skill profile in `ApplicationBuildContext`, `skill_wiring.py`, legal host | **Done** | High | `applications/_shared/skill_wiring.py` | Legal registry resolves skills |
| R-Skill.5 | **`AgentContract.skill_ids`** + validation against registry at register time | **Done** | High | `intergrax/contracts/`, `AgentRegistry` | Unknown skill_id → register error |
| R-Skill.6 | **`docs/architecture/SKILLS.md`** — catalog, layering diagram, import rules | **Done** | Medium | `docs/architecture/SKILLS.md`, `docs/README.md` index row | Approved index entry |
| R-Skill.7 | **Scaffold `new-skill`** | **Done** | Medium | `intergrax/scaffold/new_skill.py` | `python -m intergrax.scaffold new-skill <id>` |
| R-Skill.8 | **`CursorSkillImporter`** — parse `SKILL.md` + frontmatter → `SkillManifest` (best-effort; reject on schema fail) | **Done** | High | `intergrax/skills/importers/cursor_skill_md.py` | Fixture test with sample SKILL.md |
| R-Skill.9 | **Pilot skill pack** — `legal.contract_review` (tool_ids + prompt refs + policy fragment) | **Done** | High | `intergrax/skills/providers/legal/` | Legal agent lists `skill_ids`; gate green |
| R-Skill.10 | **Nexus trace events** — `SKILL_RESOLVED`, `SKILL_IMPORT_FAILED` | **Done** | Low | `runtime/events/context_skill_recording.py` | `record()` on register + import service |

**Skill vs tool enforcement:**

| Rule | Enforcement |
|------|-------------|
| Skill MUST NOT be a `ToolContract` | CI: no `ToolHandler` named `skill.*` without ADR |
| Skill MAY reference only registered `tool_id`s | `SkillResolver` validates against `ToolRegistry` |
| LLM tool-calling surface = **tools only** | Skills expand allow-list before run, not at invoke time |
| External skill without manifest validation | **Rejected** at import — no silent attach |

---

#### R-Context — Context engineering (Tier-1)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Context.1 | **`ContextBudgetPolicy`** — `max_chars`, `max_tokens_estimate`, `summary_tier` defaults; applied in `ContextManager.build_agent_context()` | **Done** | **Critical** | `runtime/nexus/context/context_budget.py` | Test: over-budget input trimmed |
| R-Context.2 | **Trace events** — `CONTEXT_ASSEMBLED`, `CONTEXT_TRIMMED` with before/after sizes | **Done** | High | `ContextManager` + `context_skill_recording` | Emitted when `event_bus` wired |
| R-Context.3 | **AGENT_CREATION_GUIDE** — “Context engineering” subsection links canon §28.1 | **Done** | Medium | `guides/AGENT_CREATION_GUIDE.md` Appendix G | No duplicate truth |
| R-Context.4 | **Finish unified tool path** — residual `use_rag` / `RagStep` callers → `rag.retrieve` | **Done** | High | `tool_gateway.py`, legal bridge, `context_builder.py` | Bridge uses `tool_ids`; LLM booleans sync in `LegalToolPlan` only |

---

#### R-Delegate — Graph-native delegation (subagent equivalent)

Intergrax does **not** implement Cursor-style nested harness in Phase R. **Delegation** = Nexus graph node with isolated memory namespace and bounded context assembly.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Delegate.1 | **`DelegationSpec` on `ExecutionNode`** — `child_agent_id`, `isolated_memory_namespace`, `context_assembly_override` | **Done** | High | `contracts/delegation.py`, `execution_graph.py` | Schema + validation |
| R-Delegate.2 | **Memory namespace isolation** — child reads/writes under `task_id/delegation/{node_id}/` via `MemoryView` | **Done** | High | `delegation_memory.py`, UAEP | Unit test |
| R-Delegate.3 | **Trace linkage** — `parent_run_id`, `parent_node_id` on child run metadata | **Done** | Medium | `graph_executor.py` | Request metadata on child node |
| R-Delegate.4 | **Integration tests** — two-agent graph with delegation node | **Done** | Medium | `test_graph_executor_delegation.py` | Gate |

---

#### R-Policy — Unified policy bundle (Tier-1 + Tier-3)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Policy.1 | **`RuntimePolicyBundle`** — aggregates tool, memory, budget, HITL, plan-loop; optional `domain_fragments: dict[str, Any]` | **Done** | High | `runtime/policy/policy_bundle.py` | Import via `policy_bundle` module (not `policy.__init__`) |
| R-Policy.2 | **Tier-3 composition** — lab/product factories build bundle once per app | **Done** | High | `policy_wiring.py`, lab/legal `wiring.py` | `ApplicationBuildContext.policy_bundle` |
| R-Policy.3 | **Canon §42.11.5** — “how to read policy for a run” operator section | **Done** | Medium | Architecture §42.11.5 | Operator runbook table |

---

#### Phase R — Definition of done

1. R row **Done** with date in Appendix E paydown log.
2. **Gate:** `uv run pytest -m gate -q` green.
3. **Skills:** at least one first-party skill pack + one importer test (R-Skill.8 or Won't fix with reason).
4. **No** new `ToolContract` entries that represent multi-step business workflows without ADR.
5. Update Appendix E status.

---

#### Phase R — Recommended execution order

```text
Wave R0 (canon):           R.0.1 → R.0.2 → R.0.3 → R.0.4
Wave R1 (skill core):      R-Skill.1 → R-Skill.2 → R-Skill.3 → R-Skill.5 → R-Skill.4
Wave R2 (skill ecosystem): R-Skill.8 → R-Skill.7 → R-Skill.9 → R-Skill.6 → R-Skill.10
Wave R3 (context):         R-Context.1 → R-Context.2 → R-Context.4 → R-Context.3
Wave R4 (delegate):        R-Delegate.1 → R-Delegate.2 → R-Delegate.3 → R-Delegate.4
Wave R5 (policy):          R-Policy.1 → R-Policy.2 → R-Policy.3
```

**Gate before Phase K.1/K.2 scale:** **Met** — Q+ **Done**, R-Skill.1–R-Skill.5 and R-Context.1 **Done**.

---
