---
id: IJ-2026-06-12-001
date: 2026-06-12
tiers:
  - tier-0
scope: EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
plan_ref:
  - D.4
status: completed
commit: pending
adr: none — repository hygiene; no architecture contract change
---

# Remove platform notebooks directory

## Operator request

Audit whether root-level platform notebooks add value or clutter the Harness architecture. Decision: remove them entirely; keep agent-local notebooks only when an agent needs interactive demos.

## Summary

Deleted the entire `notebooks/` tree (~35 `.ipynb` files: nexus demos, rag presentations, langgraph legacy, experiments templates). Updated plan docs (D.4) to point at `intergrax.experiments.workflow.ExperimentSession` and `tests/unit/experiments/` instead of notebook templates. Cleaned `.gitignore`, `bundle_intergrax_engine.py`, and `.env.example` references.

## Project impact

Platform repo surface matches Harness canon: tests and `ExperimentSession` API are the laboratory workflow source of truth; no stale `RuntimeEngine` / `rag.answers` / `ToolsAgent` demos at repo root.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` |
| Plan | `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` D.4 |
| ADR | none — hygiene only |

## Changed artifacts

- `notebooks/` — removed (entire directory)
- `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — D.4 row
- `docs/plan/PLATFORM_FOUNDATION.md`, `ORCHESTRATION.md`, `CRITIC_VERIFICATION.md`, `OBSERVABILITY.md` — D.4 sections
- `docs/guides/EXTENSION_AUTHOR_GUIDE.md`, `HARNESS_ENVIRONMENT.md` — stale notebook references
- `.gitignore`, `.env.example`, `pyproject.toml`, `tools/bundle_intergrax_engine.py` — cleanup
- `intergrax/experiments/workflow.py` — docstring

## Verification

```bash
uv run pytest tests/unit/experiments/ -q
```

Result: pass (7 passed).

## Risks and follow-ups

- `tests/fixtures/documents/md/project_structure.md` still lists removed notebooks (RAG fixture snapshot; regenerate separately if needed).
- `intergrax/legacy/rag_answers/` has no supported consumers — candidate for removal.
- Agent notebooks under `agents/<slug>/notebooks/` retained per AGENT_CREATION_GUIDE.
