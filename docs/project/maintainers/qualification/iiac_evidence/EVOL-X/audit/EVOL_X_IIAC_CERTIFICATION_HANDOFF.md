# EVOL-X — certification handoff (Session C → certification-governance-agent)

**From:** `independent-quality-auditor` (Session C) · **To:** a separate `certification-governance-agent` session (must differ from the implementer and from this auditor)
**Audit verdict handed over:** `EVOL-X — INDEPENDENT IIAC AUDIT PASS` (manifest `PASS_SCOPED`)

The auditor does not self-certify and does not publish closure. Everything below is the certifier's work.

## 1. What the certifier receives

| Artifact | Path (under `docs/project/maintainers/qualification/`) |
|---|---|
| Pass record | `evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` (byte-identical copy: `iiac_evidence/EVOL-X/audit/`) |
| Audit dossier | `iiac_evidence/EVOL-X/audit/IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md` |
| Findings register | `iiac_evidence/EVOL-X/audit/EVOL-X_audit_findings.json` |
| Scope manifest | `iiac_evidence/EVOL-X/audit/audit_scope.json` |
| Test matrix | `iiac_evidence/EVOL-X/audit/test_matrix.txt` |
| FRZ-EVOL criterion matrix | `iiac_evidence/EVOL-X/audit/frz_evol_acceptance_matrix.{json,md}` |
| Part VII 105-row matrix | `iiac_evidence/EVOL-X/audit/part_vii_id_matrix.{json,md}` |
| Manifest draft | `iiac_evidence/EVOL-X/audit/iiac_audit_session_manifest.EVOL-X.auditor_draft.json` |
| Session handoff record draft | `iiac_evidence/EVOL-X/audit/IIAC_SESSION_HANDOFF_RECORD_EVOL_X_SESSION_C.md` |
| Evidence and digests | `iiac_evidence/EVOL-X/audit/evidence/**`, `EVIDENCE_INDEX.json` |
| Record inventory | `iiac_evidence/EVOL-X/audit/evol_x_record_inventory.txt` |

All files are **untracked** in the working tree. The auditor's role registry entry has `may_commit: false`; publication is the operator's or certifier's act.

## 2. Digests (SHA-256, byte-for-byte as written, LF)

| Artifact | SHA-256 |
|---|---|
| Dossier (`audit_report_digest`) | `88f9f427d52a70a2e703600261f48b2e6d1a42689f611dde5eb6ebe3208177c6` |
| `audit_scope.json` (`audit_scope_manifest_digest`) | `24d6bcb16a011f30f10dd3e2757ba7033e328d71559914421f4d9ad4ff9da9ed` |
| `test_matrix.txt` (`test_matrix_digest`) | `4ec657686deeff0230165737a356e9e7289403d884c1f19d7b2b20c767097d3e` |
| `EVOL-X_audit_findings.json` (`findings_digest`) | `d0e7b5749e7b2ce006b69dad7cfc57ccc79d54a5a2881d559952c4000de1e95f` |
| `frz_evol_acceptance_matrix.json` | `bc9f6793f73b8fe0074dcafbb3484efc2aeb7ed2b0faaac38161ceb22062a6c1` |
| `part_vii_id_matrix.json` | `c1bb37b446dd7cd78dea33a6f115e2a10aa138807aa3146a86c1eec82db2dbad` |

Digests are computed over git **blob** bytes. `iiac_evidence/**` is `-text` in `.gitattributes`, so working-tree bytes equal blobs. The copy of the pass record under `evol_x/` is **not** under `-text`: verify it against the copy in `iiac_evidence/EVOL-X/audit/` by blob, not by CRLF working-tree bytes (F-C-10).

## 3. Exact SHA evidence the certifier may pin (it must not re-derive different values)

`audited_implementation_sha = test_execution_sha = 3c4a641b3e48a13f437e830a0562e2a80b03aabb` · `evidence_publication_sha = audit_execution_sha = bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` · audit session HEAD `06a459afbab3aa8cea89e4118865d55e7b537a34` · CI run `41337077334`.

## 4. Preconditions to complete before any publication

| # | Action | Why |
|---|---|---|
| 1 | Operator completes and signs `IIAC_SESSION_HANDOFF_RECORD_EVOL_X_SESSION_C.md` (new chat, role registry, session ids, scope) | mandatory handoff record; implementer ids are not in the repository (F-C-09) |
| 2 | Certifier fills `implementation_session_id`, `certifier_session_id`, `certification_publication_sha`; removes `manifest_status`; publishes the manifest as `evol_x/*iiac*manifest*.json` | draft fields are `PENDING` on purpose |
| 3 | Run `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py --strict-product-diff` on the completed manifest | validators passed on a test-only completed copy with `audit_execution_sha = FINAL`; they **fail** with `audit_execution_sha = HEAD` because 34 unrelated ENT-AUDIT-X product files changed after TES (F-C-08, `evidence/logs/validator_runs.txt`) |
| 4 | Reconcile `scripts/qualification/evol_x_closure_publication_preflight.py` with `IIAC_REQUIREMENT_MIGRATION_MATRIX.md` under a ticket: it still demands OD-EVOL-11 `auditor_id`, OD-EVOL-12, OD-EVOLX-08 and `b6_acceptance_publication_sha` (`evidence/logs/closure_preflight_before.txt`) | the matrix maps those gates to the certifier; the preflight also accepts a pass record or manifest by **filename glob alone**, so the certifier must validate content, not presence |
| 5 | Keep the closure commit rule: the commit that first creates the pass record must not assert the formal closed state | `IIAC_CERTIFICATION_AGENT_PUBLICATION_CONTRACT.md` |
| 6 | Publish `B6_ACCEPTANCE_PUBLICATION_SHA` and `FORMAL_CLOSURE_PUBLICATION_SHA` on the ladder in **dedicated** publication commits, with `Ref: governance/tickets/T-IGX-<NN>-<L>.md` on commits touching `scripts/` | ladder rules, ticket-first governance |

## 5. Promotion text the certifier may use (one row at a time)

Use the per-row wording and limitations from `frz_evol_acceptance_matrix.md`. Mandatory footnotes:

- FRZ-EVOL-06: `EVOL_CONSERVATIVE_FULL scope only; not PLATFORM_CONSERVATIVE_FULL` (compensates F-C-02).
- FRZ-EVOL-07: `qualification-layer deploy/rollback safety; production ActivationService.rollback durable-state enforcement is NOT claimed (OD-EVOL-05b Option 1)`.
- FRZ-EVOL-08: `change-time diagnostics; incident/deploy correlation owned by ENT-AUDIT-X; M-09H not measured (OD-EVOL-13)`.
- FRZ-EVOL-10: `executable triggers T1..T9; scenario-wave audit owned by ENT-AUDIT-X`.

A global FRZ PASS, a PROD-Q claim or a production-rollback claim is **not** supported by this audit.

## 6. Downstream cautions (ENT-AUDIT-X, ARCH-FREEZE, SCENARIO-READINESS-X)

- Do not use EVOL-X impact/selection as platform-wide test selection (LIM-01, LIM-03).
- RC-7 (`tests/unit/agents/test_reference_harness_strict.py`) still fails to collect at HEAD (F-C-11); EVOL-X does not certify a clean platform collect.
- EVOL-X evidence is bound to TES/FINAL. A material change to EVOL-X scope files after TES (none up to `06a459afb`) requires requalification per FRZ-EVOL-10.

## 7. Target outcome (not asserted by this handoff)

The intended stage outcome string is defined in `IIAC_SESSION_C_EVOL_HANDOFF_POST_OD01.md`. It may be asserted only by the authorized certification publication, after items 1–6 above.

## 8. Optional implementer corrections (separate loop; invalidate the audited SHA if applied)

F-C-01 (`replay_outcomes.get(n)` in `verify_mode_b`), F-C-03 (test-side helper/conftest TESTED_BY), F-C-04 (non-self-surface M-03 arcs), F-C-07 (registry row for `capability_edge_catalog.json`). None blocks the scoped verdict. Any change to audited code needs a new audited SHA and a re-audit of the changed part.
