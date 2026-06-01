# Intergrax Skill Library

**Last updated:** 2026-06-01 · Phase R MVP **Done** · Phase S platform `harness.*` bundle **Done**

Composable **capability packs** between the [Tool Library](TOOLS.md) and Tier-2 agents. Canon: [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.8 · Harness terms: §5.3 · Tracker: [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Appendix E.

---

## Four-layer stack

```text
Integration  →  vendor backend (Postgres, Bing, Jira)
Tool         →  atomic LLM/MCP operation (rag.retrieve)
Skill        →  tool_ids + prompt refs + policy fragment
Agent        →  UAEP module with skill_ids[] on AgentContract
```

Skills are **not** invoked by the LLM. The runtime **resolves** them into `allowed_tools` and metadata before execution.

**Unified RAG path (R-Context.4 — Done):** Prefer catalog tool `rag.retrieve` in resolved `allowed_tools` / `tool_ids`. `RuntimeToolGateway` capability plans use `tool_ids` first; legal bridge passes `tool_ids` only. `LegalToolPlan.use_rag` remains for LLM structured output and syncs to `tool_ids` via Pydantic validator — not passed to Nexus. Legacy metadata `use_rag` still honored in `ContextBuilder` for older callers.

---

## SkillManifest

| Field | Purpose |
|-------|---------|
| `skill_id` | Stable id (`legal.contract_review`) |
| `version` | Semver string |
| `description` | Human + planner readable |
| `tool_ids` | Required catalog tools |
| `prompt_instruction_ids` | Prompt Registry refs |
| `policy_fragment_id` | Optional governance fragment |
| `risk_tier` | `low` … `critical` |

Package: `intergrax/skills/core/contracts.py`

---

## Registry and profile

Mirror of the tool catalog pattern:

```python
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills

register_default_skills()
registry = build_registry_from_profile(SkillProfile(enabled_bundles=["legal"]))
```

Tier-3 helper: `intergrax.applications._shared.skill_wiring.build_application_skill_wiring`.

---

## Agent composition

```python
AgentContract(
    id="legal",
    skill_ids=["legal.contract_review"],
    allowed_tools=["rag", "websearch"],  # extras merged at register
)
```

Register with skill validation:

```python
registry.register(agent, skill_registry=skill_registry, tool_registry=tool_registry)
```

---

## External skills (Cursor SKILL.md)

```python
from intergrax.skills.importers import CursorSkillImporter

manifest = CursorSkillImporter().import_file(path)
registry.register(manifest)
```

Invalid files raise `CursorSkillImportError` — no partial attach. Use `import_cursor_skill_file(..., event_bus=bus)` to record `SKILL_IMPORT_FAILED` on the runtime event bus.

---

## Scaffold

```bash
python -m intergrax.scaffold new-skill legal.my_skill --domain legal
```

Register the bundle in `intergrax/skills/registry/bootstrap.py`.

**Application profiles:** `lab_application` enables `harness` + `legal` + `research`; `legal_application` → `legal`; `research_application` → `research` (see `skill_wiring.py`, [HARNESS_ENVIRONMENT.md](HARNESS_ENVIRONMENT.md)).

---

## First-party skills

| skill_id | Bundle | Status |
|----------|--------|--------|
| `harness.tool_smoke` | `harness` | **Done** (Phase S) |
| `harness.context_demo` | `harness` | **Done** (Phase S) |
| `harness.trace_read` | `harness` | **Done** (Phase S) |
| `legal.contract_review` | `legal` | **Done** |
| `research.literature_scan` | `research` | **Done** |

---

## Rules

- Do **not** model skills as `ToolContract`.
- Do **not** import integrations from skill code — reference `tool_id`s only.
- LLM tool-calling surface remains **tools** only.
