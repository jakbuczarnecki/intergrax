# EVOL-X — Session C independent audit evidence

Byte-for-byte audit artifacts of the independent IIAC audit of EVOL-X (`iiac_evidence/**` is `-text`). This directory is **not** an activation record, a certification record or a closure record.

| Item | State |
|---|---|
| Audit verdict | `EVOL-X — INDEPENDENT IIAC AUDIT PASS` (manifest `PASS_SCOPED`) |
| Manifest | `iiac_audit_session_manifest.EVOL-X.auditor_draft.json` — **draft**, certifier and implementer fields pending |
| Certification publication | not done (separate `certification-governance-agent`) |
| EVOL-X closed | **no** |
| FRZ-EVOL rows promoted | **none** |

Start with `IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md`, then `EVOL_X_IIAC_CERTIFICATION_HANDOFF.md`. Digests of every file: `EVIDENCE_INDEX.json`.

| Path | Content |
|---|---|
| `IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md` | audit dossier |
| `EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` | pass record (copy also placed under `evol_x/`) |
| `EVOL_X_IIAC_CERTIFICATION_HANDOFF.md` | handoff to the certifier |
| `EVOL-X_audit_findings.json` | findings F-C-01..F-C-11 with ECAP §13 fields |
| `frz_evol_acceptance_matrix.{json,md}` | FRZ-EVOL-01..10 criterion-level matrix |
| `part_vii_id_matrix.{json,md}` | master plan Part VII, 105 rows |
| `audit_scope.json`, `test_matrix.txt` | scope and executed matrix |
| `iiac_audit_session_manifest.EVOL-X.auditor_draft.json` | manifest draft |
| `IIAC_SESSION_HANDOFF_RECORD_EVOL_X_SESSION_C.md` | operator attestation draft |
| `evol_x_record_inventory.txt` | EVOL-X record and code inventory with git blob SHAs at HEAD |
| `evidence/logs/` | replay logs, JUnit XML, probe outputs, CI excerpt, validator runs |
| `evidence/probes/` | probe and replay scripts (final versions) |
