# EVOL-X E2 — Final evidence assembly (EVIDENCE_FINAL)

**Trigger:** Genuine human M-09H measurements + AC-M09H-01 pass. **Do not** run this block with template placeholders.

## Target artifacts @ `evol_x_p03_independent_replay/`

| Artifact | Action |
|----------|--------|
| `evol_m09_drill_case_registry.v1.json` | Replace with [rebind_staging](./artifacts/evol_m09_drill_case_registry.v1.rebind_staging.json) bytes |
| `evol_m09_human_drill.v2.json` | New — from worksheet (strict v2 model) |
| `evol_m09_human_drill_worksheet.v1.json` | New — genuine human fields |
| `evol_metrics.v1.json` | Update `EVOL-M-09H` → numerator/denominator 3/3, `status: MEASURED`, `passed: true`, `population_digest` from rebound registry |
| `MANIFEST.json` | Add artifact paths for registry (if not listed), drill, worksheet |
| `ARTIFACT_DIGESTS.json` | Recompute SHA-256 for every bundled file @ FINAL HEAD |

## Build `evol_m09_human_drill.v2.json`

Required fields: see [EVOL_X_M09H_HUMAN_DRILL_RUNBOOK.md](../EVOL_X_M09H_HUMAN_DRILL_RUNBOOK.md) and `M09HumanDrillArtifact` in `testing_support/change_safety/contracts/artifacts.py`.

| Field | Source |
|-------|--------|
| `schema_version` | exactly `evol_m09_human_drill.v2` |
| `measurement_policy_id` | exactly `OD-EVOL-08-M1` |
| `source_commit_sha` / `test_execution_sha` | `dcd7ae6e264d435994c61162e0d41aba25f71283` |
| `registry_digest` | SHA-256 of committed rebound registry **bytes** |
| `cases[]` | Genuine `tool_seconds`, `attribution_seconds`, `attributed_cause_identities`, `correct` |
| `median_*` | Medians of case values (±0.001) |
| `adversarial_control_passed` | true only if AC-M09H-01 pass |
| `drill_*_at_utc` | ISO-8601 with timezone |
| `evaluator_*` | Non-empty |
| `worksheet_digest` | SHA-256 of committed worksheet bytes |

## Worksheet digest

```powershell
$ws = "docs/project/maintainers/qualification/evol_x/evol_x_p03_independent_replay/evol_m09_human_drill_worksheet.v1.json"
uv run python -c "from pathlib import Path; from testing_support.change_safety.evidence import p03; print(p03.sha256_hex(Path('$ws').read_bytes()))"
```

## Mode A verification (final)

Clean tree @ FINAL commit:

```powershell
uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish
```

Expect: `EVIDENCE_INTEGRITY_OK`, disposition `final`, no `--technical`.

## Mode B (post-FINAL auditor)

```powershell
git checkout dcd7ae6e264d435994c61162e0d41aba25f71283
uv run pytest tests/qualification/evol_x -q --junitxml=.tmp/session/evol-x-e2-m09h/junit.xml
uv run python scripts/qualification/evol_x_p03_evidence.py verify-test-execution `
  --evidence-ref <EVIDENCE_FINAL_PUBLICATION_SHA> `
  --certification --replay-junit .tmp/session/evol-x-e2-m09h/junit.xml
```

## Publication

1. Commit only evidence bundle + required doc pointer updates on `development`.
2. Record `EVIDENCE_FINAL_PUBLICATION_SHA` in [EVOL_X_SHA_LADDER.md](../EVOL_X_SHA_LADDER.md) (operator edit — not done in prep commit if ladder frozen).
3. Push `origin/development`.
4. Hand off to B6 per [EVOL_X_B6_INDEPENDENT_AUDIT_PACKAGE.md](../EVOL_X_B6_INDEPENDENT_AUDIT_PACKAGE.md).

## Explicit prohibitions

- No synthetic timings or PASS placeholders
- No Mode A final against TECH-only HEAD for closure
- No change to TES, TECH, W6-04 CI record, or FRZ statuses in prep-only commits
