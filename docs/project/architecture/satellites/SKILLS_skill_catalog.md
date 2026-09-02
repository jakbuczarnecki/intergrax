# SKILLS - skill catalog

**Parent hub:** [`SKILLS.md`](../SKILLS.md)

## First-party catalog (149 skills · 41 bundles)

| Bundle | skill_ids | Status |
|--------|-----------|--------|
| Bundle | Skills (count) | Status |
|--------|----------------|--------|
| `harness` | 14 | **Done** |
| `rag` | 8 | **Done** |
| `ops` | 12 | **Done** |
| `legal` | 6 | **Done** |
| `research` | 7 | **Done** |
| `eval` | 6 | **Done** |
| `memory` | 6 | **Done** |
| `platform` | 8 | **Done** |
| `dev` | 6 | **Done** |
| `data` | 6 | **Done** |
| `modality` | 5 | **Done** |
| `collaboration` | 5 | **Done** |
| `workspace` | 4 | **Done** |
| `hitl` | 4 | **Done** |
| `storage` | 4 | **Done** |
| `message_bus` | 4 | **Done** |
| `knowledge` | 3 | **Beta** |
| `graph` | 3 | **Done** |
| `sandbox` | 3 | **Done** |
| `cache` | 3 | **Done** |
| `notify` | 3 | **Done** |
| `health` | 3 | **Done** |
| `cost`, `identity`, `filesystem`, `agent` | 2 each | **Done** |
| `catalog`, `cloud_platform`, `code`, `http`, `jira`, `gitlab`, `ml`, `openai`, `context`, `vector_store`, `crm`, `billing`, `metrics`, `browser`, `interaction` | 1–2 each | **Done** |

**SK-EXP5 (2026-06-08):** +50 compositional packs - product verticals (legal ops, on-call SRE, research lab, data platform, sandbox dev) without new bundles.

Per-skill `USAGE.md` under `intergrax/skills/providers/<bundle>`.
Verify: `register_default_skills()` → **149** · gate: `test_sk_exp5_skill_bundles.py`, `test_skill_usage_docs.py`.

---

## Rules

- Do **not** model skills as `ToolContract`.
- Do **not** import integrations from skill code - reference `tool_id`s only.
- LLM tool-calling surface remains **tools** only.
- Skills expand allow-lists **before** run - not at LLM invoke time.
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
