# EVOL-X B6 — Mode A provenance record

**Evidence HEAD:** `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1`  
**TES (manifest):** `3c4a641b3e48a13f437e830a0562e2a80b03aabb`

## Command (canonical)

```powershell
git checkout bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1
uv sync --extra dev-ci --frozen
uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish
```

## Session execution

| Field | Value |
|-------|-------|
| Log | `.tmp/session/evol-x-b6-multi/logs/a2-mode-a.log` |
| Result line | `{"status": "EVIDENCE_INTEGRITY_OK", "mode": "A", "disposition": "final", "evidence_head": "bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1"}` |
| Exit code | 0 (implicit with OK status in session) |

## Observed bundle facts @ FINAL

- `certification_profile`: `AUTOMATION_FIRST_SCOPED`
- `evol_m09_automated_diagnostic_qualification.v1.json` present; human drill absent
- `EVOL-M-09H`: NOT_MEASURED; `EVOL-M-09A`: MEASURED PASS in `evol_metrics.v1.json`

## Disposition

**G-05 technical PASS** — pending formal auditor attestation in G-07 record.
