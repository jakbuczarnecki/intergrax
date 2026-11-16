# IIAC certifier handoff — ENT-AUDIT-X W4 Batch D → W5

**Stage:** `ENT-AUDIT-X` · **Policy:** `IIAC-2026-11-05.1`  
**Date (UTC):** 2026-11-11 · **Ticket:** `governance/tickets/T-IGX-024-B.md`

## Session attestation

- [x] New sub-agent chat (`certification-governance-agent`); **not** implementer; **not** auditor `7c58500c-17c9-42a8-acbc-4b0e05487751`
- [x] `implementation_session_id`: `T-IGX-024-B-R-FRESH-R-PUB__cursor-a86f208e-20e4-4f4b-b233-bfef6bd713e7`
- [x] `auditor_session_id` (W3 publication): `7c58500c-17c9-42a8-acbc-4b0e05487751`
- [x] `certifier_session_id`: `4380025f-0a4e-43a2-bb90-ead2ce6d0bec`
- [x] SHA lock: TES `32a963344…` · EVID `481b2cc2…` · FREEZE pub `bc4b5f61…` · W3 `audit_execution_sha` `dc778d12…` · baseline `480550d80…`

## W4 Batch D outputs (this publication)

| Artefact | Path |
| --- | --- |
| Certification manifest | `iiac_audit_session_manifest.ENT-AUDIT-X.w4_dc778d.json` |
| Certification record | `ENT_AUDIT_X_W4_IIAC_CERTIFICATION_RECORD.json` |
| Certifier report | `ENT_AUDIT_X_W4_CERTIFIER_REPORT.md` |
| FRZ-AUD table | `ENT_AUDIT_X_W4_FRZ_AUD_CERTIFIER_VERDICT_TABLE.md` |
| Pass record | `ENT_AUDIT_X_W4_IIAC_CERTIFICATION_PASS_RECORD.md` |
| W3 audit input | `../audit/w3_dc778d/` (unchanged) |

## Validators executed @ publication worktree

| Command | Result |
| --- | --- |
| `verify_iiac_policy_activation.py --require-in-force` | PASS (IN_FORCE) |
| `verify_iiac_audit_manifest.py` (W3 manifest) | PASS |
| `verify_iiac_role_separation.py` (W3 manifest) | PASS |
| `verify_od_ent_02_compensating_control.py` | PASS (package draft) |

## W5 STOP contract (not started here)

1. **W4-03** — dedicated commits promoting FRZ-AUD-01..10 rows + Matrix C overlay reconcile (certifier/implementer lane).
2. **Gate A** — `ent_audit_x_formal_closure_preflight.py --gate A` green (D-3 preflight work).
3. **OD-ENT-08** — operator signing session only; **no** OD-ENT-08 in this batch.
4. **Closure phase 1/2** — roadmap §5 headline, `FORMAL_CLOSURE_PUBLICATION_SHA`, Gate B — W5.
5. **No CLOSED** headline in this publication.

## Operator signature line

`Approved handoff: operator @ pending-orchestrator-review`
