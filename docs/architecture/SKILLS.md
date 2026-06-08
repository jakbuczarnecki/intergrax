# Skills

**Status:** Canonical architecture (domain pair 1:1)  
**Last updated:** 2026-06-08 — SK-EXP through SK-EXP4 shipped; **99** skills · **41** bundles  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/SKILLS.md`](../plan/SKILLS.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) **Appendix J**  
**Audit layers:** 12  

---

## Four-layer stack

```text
Integration  →  vendor backend (Postgres, Bing, Jira)
Tool         →  atomic LLM/MCP operation (rag.retrieve)
Skill        →  tool_ids + prompt refs + policy fragment
Agent        →  UAEP module with skills[] on AgentContract
```

Skills are **not** invoked by the LLM. The runtime **resolves** them into `allowed_tools` and metadata at **agent registration** (and via the diagnostic catalog tool `skill.resolve`).

**Unified RAG path (R-Context.4 — Done):** Prefer catalog tool `rag.retrieve` in resolved `allowed_tools` / `tool_ids`. `RuntimeToolGateway` capability plans use `tool_ids` first; legal bridge passes `tool_ids` only. `LegalToolPlan.use_rag` remains for LLM structured output and syncs to `tool_ids` via Pydantic validator — not passed to Nexus. Legacy metadata `use_rag` still honored in `ContextBuilder` for older callers.

---

## Skill engine — two registry layers

Intergrax mirrors the tool catalog pattern: a **static catalog** (bootstrap metadata) and a **runtime registry** (lookup by `skill_id`).

```text
Tier-0 catalog (process-wide, idempotent bootstrap)
  register_default_skills() / SkillPlugin / EP intergrax.skills
        → SkillCatalog (_CATALOG: SkillBundleEntry per bundle_id)
        → register_skill_bundle(entry)

Tier-3 / agent runtime
  SkillProfile (enabled_bundles | enabled | register_all_catalog_bundles)
        → build_registry_from_profile(profile)
        → SkillRegistry (_skills: skill_id → RegisteredSkill)

Agent bind time (Tier-1)
  AgentContract.skills[] + extra_tools[]
        → SkillResolver.resolve_skills()
        → resolve_contract_tools()
        → AgentContract.allowed_tools (resolved)
        → SKILL_RESOLVED event (optional)
```

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Catalog | `skills/registry/catalog.py` | Bundle metadata, `iter_bundles()`, `list_catalog_skill_ids()` |
| Bootstrap | `skills/registry/bootstrap.py` | `register_default_skills(bundle_ids=…)` — shipped plugins |
| Plugin wire | `skills/registry/plugin_register.py` | `register_skill_plugin()` → catalog + register fn |
| Runtime registry | `skills/registry/runtime.py` | `SkillRegistry.register(manifest)` |
| Profile factory | `skills/registry/factory.py` | `build_registry_from_profile(SkillProfile)` |
| Resolver | `skills/resolver.py` | `SkillResolver` / `SkillResolverProtocol` → `ResolvedSkillPack` |
| Contract merge | `skills/integration/contract_resolution.py` | `resolve_contract_tools()` |
| Agent bind | `runtime/registry/agent_registry.py` | Merge skills at `register()` |
| Tier-3 wiring | `applications/_shared/skill_wiring.py` | `build_application_skill_wiring()` |
| Tool auto-enable | `applications/_shared/skill_tool_profile.py` | `extend_tool_profile_for_skills()` |
| Runtime snapshot | `applications/_shared/catalog_runtime_bridge.py` | `RuntimeConfig.skill_profile` (TS-1) |

**Shipped bundles (41):** `harness`, `rag`, `workspace`, `memory`, `research`, `knowledge`, `legal`, `ops`, `dev`, `browser`, `collaboration`, `data`, `platform`, `sandbox`, `hitl`, `graph`, `storage`, `message_bus`, `cache`, `eval`, `modality`, `notify`, `cost`, `identity`, `health`, `context`, `agent`, `vector_store`, `crm`, `billing`, `metrics`, `catalog`, `cloud_platform`, `code`, `filesystem`, `http`, `interaction`, `jira`, `gitlab`, `ml`, `openai` — registered via `skills/registry/shipped_plugins.py`.  
`knowledge` remains **BETA**; all other shipped bundles **STABLE**.

---

## Tier-3 host pipeline

Canonical entry: `wire_application_environment()` (`applications/_shared/environment_wiring.py`).

```text
wire_application_environment(manifest, env)
  ├── bootstrap_application_integration_catalog()
  ├── tool_profile = tool_profile_with_sandbox(env)
  ├── tool_profile = extend_tool_profile_for_skills(tool_profile, env.skill_profile)
  ├── build_application_tool_wiring(tool_profile, …)     → ToolRegistry
  ├── build_application_skill_wiring(env.skill_profile)    → SkillRegistry
  ├── wire_policy_bundle(env)
  ├── resolve_prompt_registry(env.prompt_profile)
  ├── ApplicationBuildContext (profiles + registries)
  ├── EnvironmentSkillToolConsistencyCheck (roster ⊆ environment)
  └── catalog_runtime_bridge → RuntimeConfig.skill_profile
```

**Rule:** enable skills on `ApplicationEnvironmentProfile.skill_profile` — do not create agent-local skill registries. See Appendix J.

**Presets** (`skill_wiring.py`, SK-PRESET.1):

| Helper | `enabled_bundles` |
|--------|-------------------|
| `harness_platform_skill_profile()` | `harness` |
| `lab_skill_profile()` | `harness`, `legal`, `research`, `rag`, `workspace`, `memory`, `knowledge` |
| `research_skill_profile()` | `research`, `rag`, `browser` |
| `legal_skill_profile()` | `legal`, `rag`, `knowledge`, `workspace` |
| `knowledge_skill_profile()` | `knowledge` |
| `rag_skill_profile()` | `rag` |
| `ops_skill_profile()` | `ops`, `dev`, `workspace` |
| `platform_skill_profile()` | `platform`, `rag`, `memory`, `research` |
| `lkw_skill_profile()` | `rag`, `workspace`, `memory`, `knowledge` |
| `dispute_skill_profile()` | `legal`, `rag`, `memory`, `research` |

---

## SkillManifest

| Field | Purpose |
|-------|---------|
| `skill_id` | Stable id (`legal.contract_review`) |
| `version` | Semver string |
| `description` | Human + planner readable |
| `tool_ids` | Required catalog tools (unique within manifest) |
| `prompt_instruction_ids` | Prompt Registry refs (not inline prompt blobs) |
| `policy_fragment_id` | Optional governance fragment id |
| `risk_tier` | `low` · `medium` · `high` · `critical` |
| `tags` | Discovery / governance labels |
| `requires_skills` | Other `skill_id`s merged **before** this skill (transitive; cycle → error) |

Package: `intergrax/skills/core/contracts.py`  
Plugin protocol: `intergrax/skills/core/plugin.py` (`SkillPlugin`, entry point `intergrax.skills`)

---

## SkillResolver and ResolvedSkillPack

`SkillResolver` performs **pure registry lookups** — no LLM calls.

```python
from intergrax.skills.resolver import SkillResolver, ResolvedSkillPack

pack: ResolvedSkillPack = SkillResolver(skill_registry, tool_registry).resolve(
    ["research.literature_scan"]
)
# pack.skill_ids      — expanded order (requires_skills first)
# pack.tool_ids       — frozenset union
# pack.prompt_instruction_ids
# pack.policy_fragment_ids
# pack.risk_tier      — max tier across merged skills
```

| Behavior | Detail |
|----------|--------|
| `requires_skills` | Topological expansion; cycle raises `SkillResolutionError` |
| Tool validation | When `tool_registry` is provided, every `tool_id` must exist |
| Unknown skill | `SkillResolutionError` at resolve / validate |
| `merged_allowed_tools(extra)` | Union `pack.tool_ids` with `extra_tools` tool_ids |

Typed contract: `SkillResolverProtocol` (Phase TS-3).

---

## Agent composition

Agents declare **manifest objects**, not string ids:

```python
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.skills.providers.legal.manifests import LEGAL_CONTRACT_REVIEW

AgentContract(
    id="legal",
    skills=[LEGAL_CONTRACT_REVIEW],
    extra_tools=[],  # optional ToolContract refs beyond skill union
    # allowed_tools is OUTPUT — set by AgentRegistry.register
)
```

Register with skill + tool validation:

```python
registry.register(
    agent,
    skill_registry=skill_registry,
    tool_registry=tool_registry,
    event_bus=event_bus,  # optional → SKILL_RESOLVED
)
```

### Resolution semantics (canonical)

`resolve_contract_tools()` (`skills/integration/contract_resolution.py`):

1. Validate each `SkillManifest` in `contract.skills` exists in `SkillRegistry`.
2. `SkillResolver.resolve_skills(contract.skills)` — expand `requires_skills`, merge metadata.
3. Union skill `tool_ids` with `extra_tools[].tool_id`.
4. **Replace** `contract.allowed_tools` with the merged list (pre-declared `allowed_tools` on the author contract are **not** preserved).

Environment intersection happens later via `ToolProfile` / `ToolAccessPolicy` — not inside the resolver.

If `skill_registry` is omitted at register time, `AgentRegistry` bootstraps all catalog bundles (`register_all_catalog_bundles=True`).

---

## Runtime effects — implemented today vs manifest fields

| Output | Status | Consumer |
|--------|--------|----------|
| `allowed_tools` merge | **Done** | `AgentRegistry.register`, conformance checks |
| `extend_tool_profile_for_skills` | **Done** | Tier-3 `wire_application_environment` |
| `SKILL_RESOLVED` trace | **Done** | `context_skill_recording.record_skill_resolved` |
| `SKILL_IMPORT_FAILED` trace | **Done** | `import_cursor_skill_file(..., event_bus=…)` |
| Capability graph nodes | **Done** | `capability_graph.py` (prompt + policy edges) |
| `skill.resolve` catalog tool | **Done** | Diagnostic / planner introspection (`TOOLS.md`) |
| `prompt_instruction_ids` → `ContextManager` | **Planned** | Track **SK-BRIDGE.1** in plan — agents consume prompts in UAEP steps today |
| `policy_fragment_id` → `RuntimePolicyBundle` | **Planned** | Track **SK-BRIDGE.2** in plan — Tier-3 `domain_policy_fragments` is separate |

Until SK-BRIDGE.* ships, declare `prompt_instruction_ids` and `policy_fragment_id` for governance/traceability; wire prompts in agent steps or Prompt Registry explicitly.

---

## Registry bootstrap (standalone)

Mirror of tool catalog pattern:

```python
from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.skills.registry import SkillProfile, SkillRegistry, build_registry_from_profile

bootstrap_catalogs(register_shipped=True, skill_bundle_ids=("legal", "research"))
registry: SkillRegistry = build_registry_from_profile(
    SkillProfile(enabled_bundles=["legal", "research"])
)
```

External bundles: `SkillPlugin` + `register_skill_plugin()` or entry point `intergrax.skills`. See [guides/EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md).

---

## External skills (Cursor SKILL.md)

```python
from pathlib import Path

from intergrax.skills.importers.service import import_cursor_skill_file
from intergrax.skills.registry.runtime import SkillRegistry

manifest = import_cursor_skill_file(Path("path/to/SKILL.md"), event_bus=bus)
runtime = SkillRegistry()
runtime.register(manifest)  # does NOT add a catalog bundle — host must enable explicitly
```

Invalid files raise `CursorSkillImportError` — no partial attach. Failed imports with `event_bus` emit `SKILL_IMPORT_FAILED`.

Importer module: `skills/importers/cursor_skill_md.py` (`CursorSkillImporter`).

---

## Scaffold

```bash
python -m intergrax.scaffold new-skill legal.my_skill --domain legal
# alias: new-skill-bundle
```

Register with `register_skill_plugin(...)` or add the plugin class to `shipped_plugins.py`.

**Layout:**

```text
intergrax/skills/providers/<bundle>/
  manifests.py              # SkillManifest constants
  plugin.py                 # SkillPlugin
  bundle.py                 # register_skill_plugin() helper
  USAGE.md                  # bundle index (links to per-skill docs)
  <skill_id>/USAGE.md       # required per skill — Purpose, How it works, How to use, What you get
```

Every shipped `skill_id` **must** have a filled `intergrax/skills/providers/<bundle>/<skill_id>/USAGE.md` (English). Gate: `test_skill_usage_docs.py`.

---

## First-party catalog (99 skills · 41 bundles)

| Bundle | skill_ids | Status |
|--------|-----------|--------|
| `harness` | 12 skills incl. `harness.run_comparator`, `harness.run_exporter` | **Done** |
| `rag` | 5 skills | **Done** |
| `workspace` | 2 skills | **Done** |
| `memory` | 4 skills incl. `memory.semantic_search` | **Done** |
| `research` | 4 skills incl. `research.web_cache_admin` | **Done** |
| `knowledge` | 3 skills incl. `knowledge.confluence_navigator` | **Beta** |
| `legal` | 3 skills | **Done** |
| `ops` | 8 skills incl. `ops.log_tail`, `ops.incident_ack` | **Done** |
| `dev` | 3 skills | **Done** |
| `browser` | 2 skills incl. `browser.interactive_run` | **Done** |
| `collaboration` | 3 skills | **Done** |
| `data` | 4 skills incl. `data.sql_mutator`, `data.records_admin` | **Done** |
| `platform` | 5 skills incl. `platform.secret_admin`, `platform.workflow_cancel` | **Done** |
| `sandbox` | 1 skill | **Done** |
| `hitl` | 2 skills | **Done** |
| `graph` | 1 skill | **Done** |
| `storage` | 2 skills incl. `storage.object_lifecycle` | **Done** |
| `message_bus` | 2 skills | **Done** |
| `cache` | 2 skills incl. `cache.key_admin` | **Done** |
| `eval` | 4 skills incl. `eval.observation_browser` | **Done** |
| `modality` | 3 skills incl. `modality.vision_segment` | **Done** |
| `notify` | 2 skills incl. `notify.batch_dispatch` | **Done** |
| `cost`, `identity`, `health`, `context`, `agent`, `vector_store`, `crm`, `billing`, `metrics` | 1 skill each (SK-EXP3) | **Done** |
| `catalog` | `catalog.tool_introspect` | **Done** (SK-EXP4) |
| `cloud_platform` | `cloud_platform.resolver` | **Done** (SK-EXP4) |
| `code` | `code.runner` | **Done** (SK-EXP4) |
| `filesystem` | `filesystem.local_io` | **Done** (SK-EXP4) |
| `http` | `http.api_client` | **Done** (SK-EXP4) |
| `interaction` | `interaction.session_handler`, `interaction.input_capture` | **Done** (SK-EXP4) |
| `jira` | `jira.task_navigator` | **Done** (SK-EXP4) |
| `gitlab` | `gitlab.issue_creator` | **Done** (SK-EXP4) |
| `ml` | `ml.explain_predict` | **Done** (SK-EXP4) |
| `openai` | `openai.vector_admin` | **Done** (SK-EXP4) |

Per-skill `USAGE.md` under `intergrax/skills/providers/<bundle>/`.  
Verify counts: `register_default_skills()` → `list_catalog_skill_ids()` · gate: `test_sk_exp_skill_bundles.py` … `test_sk_exp4_skill_bundles.py`, `test_skill_usage_docs.py`.

---

## Rules

- Do **not** model skills as `ToolContract`.
- Do **not** import integrations from skill code — reference `tool_id`s only.
- LLM tool-calling surface remains **tools** only.
- Skills expand allow-lists **before** run — not at LLM invoke time.
- Prompt and evaluation governance for skill packs should follow Phase V streams:
  - prompt architecture/regression: `V-PE.*`
  - evaluation baselines/trends: `V-EVAL.*`

---

## Verification

| Concern | Command |
|---------|---------|
| Skill resolver | `uv run pytest tests/unit/skills/ -m gate -q` |
| Catalog runtime bridge | `uv run pytest tests/unit/applications/test_catalog_runtime_bridge.py -m gate -q` |
| Environment conformance | `uv run pytest tests/unit/applications/ -m gate -k conformance` |
| Full gate | `uv run pytest -m gate -q` |

Author map and control-plane diagram: Appendix J in `AGENT_CREATION_GUIDE.md`.
