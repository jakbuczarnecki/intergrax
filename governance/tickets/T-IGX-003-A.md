# T-IGX-003-A — DCP post phase 0–7 gap closure

## Scope

Proportionate follow-ups after DCP phases 0–7 (`2eddd4223`): doc cross-links to devos hub, rules README completeness, manifest/registry notes (skill count, sync-adapters, GOV-06 advisory), dedupe `AGENTS.md` DevOS section.

## Plan digest

Plan digest: (operator plan `devos_intergrax_qc_5870e73e` — procedural; no pinned file in repo)

## Approved

Approved: operator @ 2026-11-04

## Test contract

```bash
uv run pytest tests/unit/governance/test_dcp_*.py -q
uv run python -m scripts.governance.igx_gov doctor
```

## Out of scope

- Rewriting history for missing Ref on `2eddd4223`
- TRUSTED_POLICY_PIN CI job (GOV-06 deferred)
- New igx-devos skills beyond registry parity (11 intentional vs reference ~15)

## Lifecycle state

closed
