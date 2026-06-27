---
id: IJ-2026-06-12-005
date: 2026-06-12
tiers:
  - tier-0
scope: docs/adr harness ADR layout
plan_ref:
  - P-Ext.5.1
status: completed
commit: pending
adr: none — documentation layout only; no platform contract change
---

# Harness ADR date-partitioned entries layout

## Operator request

Partition growing `docs/adr/` harness ADRs into date folders (like the implementation journal), move existing files by creation date, and update scaffolds/rules so new ADRs land under `entries/YYYY-MM-DD/`.

## Summary

Introduced `docs/adr/entries/YYYY-MM-DD/` for all 21 harness ADRs, rewrote cross-repo links, added `intergrax/scaffold/harness_adr.py` path helpers, `scripts/maintenance/check_harness_adr.py` gate, and updated scaffold README/TEMPLATE plus iteration rules.

## Project impact

Harness ADR catalog scales chronologically without a flat root directory; authors and CI have a single layout contract aligned with the implementation journal.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/intergrax_runtime_architecture.md` |
| Plan | `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` P-Ext.5.1 (scaffold) |
| ADR index | `docs/adr/README.md` |
| Rules | `.cursor/rules/intergrax-iteration.mdc` |

## Changed artifacts

- `docs/adr/entries/**` — relocated harness ADRs by first-commit date
- `docs/adr/README.md`, `docs/adr/TEMPLATE.md` — layout + index
- `intergrax/scaffold/harness_adr.py`, `intergrax/scaffold/adr_templates.py` — path helpers and scaffold canon
- `scripts/maintenance/check_harness_adr.py` — INDEX ↔ filesystem gate
- `AGENTS.md`, `.cursor/rules/intergrax-iteration.mdc` — process + verification
- Cross-repo doc links (~47 files) — `entries/<day>/` paths

## Verification

```bash
uv run pytest tests/unit/scaffold/test_adr_scaffold.py tests/unit/scaffold/test_harness_adr_paths.py -q
uv run python scripts/maintenance/check_harness_adr.py
```

Result: pass (24 tests; harness ADR check OK).

## Risks and follow-ups

- Future ADR-MEM-002 and other planned ADRs must use `entries/YYYY-MM-DD/` on creation and update `docs/adr/README.md` index.
