<!-- Draft added by governance/tickets/T-IGX-044-A.md (EQO-A remediation node PT-E00, documentation only). Not adopted: the ticket carries no approval line. -->

# IIAC manifest drafts that fail the repository validators - point-in-time record (EQOA-A2-006)

Status: **DRAFT - point-in-time record, not a certificate.** It records facts about three files; it does not certify, close, supersede or re-label anything, and it grants no exception. Facts were measured at `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2` (`origin/development` when this record was written). The finding behind it is EQO-A A2-006 (baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`); the standing entry in the register is section 9.6 of [`enterprise_debt_register/README.md`](../enterprise_debt_register/README.md).

## 1. Result

Of the 40 `iiac_audit_session_manifest*.json` files under `iiac_evidence/` at the head above (39 at the baseline; the later one is `PDC-X/P8/activation/iiac_audit_session_manifest.IIAC-ACT-N11-PDC-X-P8-04.json`), **3 fail both** `scripts/governance/verify_iiac_audit_manifest.py` and `scripts/governance/verify_iiac_role_separation.py`; the other 37 pass both. All three failing files are drafts that carry the placeholder `PENDING_CERTIFIER` or lack required fields. The corpus run was one process, one validator call per file, and is logged under the session notes of this change (`manifest_corpus_run.log`); it is not a tracked artefact.

## 2. Treatment

The three files stay exactly where they are, byte for byte: they are not edited, moved, renamed or deleted, because evidence indexes and a publication note carry their digests and one script carries one path (section 4). This record is the in-place supersession-or-point-in-time entry that the remediation plan requires instead. It states what each draft is and which later manifest of the same evidence line exists; it does **not** declare that any manifest replaces any draft, because no tracked record does so (section 5).

## 3. The three drafts

| | Draft A | Draft B | Draft C |
|---|---|---|---|
| Path under `iiac_evidence/` | `EVOL-X/audit/iiac_audit_session_manifest.EVOL-X.auditor_draft.json` | `EVOL-X/b4_reaudit/iiac_audit_session_manifest.EVOL-X-B4.reaudit_draft.json` | `EVOL-X-W4-01-REAUDIT/iiac_audit_session_manifest.EVOL-X-W4-01.reaudit_draft.json` |
| `stage_id` | `EVOL-X` | `EVOL-X-B4-CORRECTIVE` | `EVOL-X-W4-01-CONSOLIDATED-B4-CORRECTIVE` |
| `verdict` in the file | `PASS_SCOPED` | `CONDITIONAL` | `CONDITIONAL` |
| `audited_implementation_sha` | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` | `23feb2bbb017b1beb63f9368b5978384093ff924` | `9f8b04d2b8b670077b1e78ce812bb34b680f3968` |
| `audit_execution_sha` | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` | `9a8fa3634713f69b56ce87ceaf9ed319fad27dae` | `9f8b04d2b8b670077b1e78ce812bb34b680f3968` |
| SHA-256 of the file | `eef9ed1b2460eff9c2d0c2ce15b7b2e77412ea3a407c71a6c5c92acfb5fc0d0d` | `83db6500965e3ed55b4d0b5a397470312f013ecadd2090f95e792cd85a43b129` | `d3ff7062e7e3d119f539b3e5cfce25bf0d8c49f3e2d4e4216f8c98b4e64bdcab` |
| Placeholder fields | `certification_publication_sha` | `evidence_publication_sha`, `certification_publication_sha` | `evidence_publication_sha`, `certification_publication_sha`; also lacks `audit_scope_manifest_digest` and `test_matrix_digest` |
| `verify_iiac_audit_manifest` | FAIL: schema pattern for `PENDING_CERTIFIER`; `certification_publication_sha` not 40-hex | FAIL: schema pattern; `evidence_publication_sha` and `certification_publication_sha` not 40-hex | FAIL: schema requires `audit_scope_manifest_digest`; both digests missing; both publication SHAs not 40-hex |
| `verify_iiac_role_separation` | FAIL: `certification_publication_sha` not 40-hex | FAIL: both publication SHAs not 40-hex | FAIL: both digests missing; both publication SHAs not 40-hex |

Both validators exit 1 on each draft (`three_drafts_validators.log`, session notes of this change).

## 4. Where the drafts are cited (digests and paths must keep resolving)

| Draft | Citation |
|---|---|
| A | `iiac_evidence/EVOL-X/audit/EVIDENCE_INDEX.json:195` (digest entry); `iiac_evidence/EVOL-X/audit/EVOL_X_IIAC_CERTIFICATION_HANDOFF.md:19`; `iiac_evidence/EVOL-X/audit/README.md:8,24`; `iiac_evidence/EVOL-X/audit/EVOL-X_audit_findings.json:155`; `scripts/qualification/evol_x_closure_publication_preflight.py:176` (`REL_AUDITOR_DRAFT`) |
| B | `iiac_evidence/EVOL-X/b4_reaudit/EVIDENCE_INDEX.json:38` (digest entry) |
| C | `iiac_evidence/EVOL-X-W4-01-REAUDIT/PUBLICATION_PROVENANCE_NOTE.md:28` (digest line, matches the SHA-256 above); `iiac_evidence/EVOL-X-W4-01-REAUDIT/EVOL_X_W4_01_REAUDIT_DOSSIER.md:89`; `evol_x/EVOL_X_W4_01_SECOND_CERTIFIER_REPORT.md:56` (records that both validators fail on the "only W4-01 manifest") |
| A, B, C | `enterprise_debt_register/README.md:120-122` (standing entry) |

The SHA-256 values of A and B equal the digests in their `EVIDENCE_INDEX.json` entries at the head above. Editing or moving a draft would break these digests and the path in the preflight script.

## 5. Later manifests of the same evidence line (observation, not a supersession declaration)

- **Draft A** and `iiac_evidence/EVOL-X/certification/iiac_audit_session_manifest.EVOL-X.json`: the two files have identical `stage_id`, `audited_implementation_sha`, `test_execution_sha`, `evidence_publication_sha`, `audit_execution_sha`, `audit_scope_manifest_digest`, `test_matrix_digest` and `verdict`; they differ in `certification_publication_sha` (placeholder in A, `ad0877715ed079ce920ff9d925fda03555b2c04e` in the certification manifest) and in the limitation list. The certification manifest passes both validators.
- **Draft B** and **Draft C**: the later EVOL-X manifest `iiac_evidence/EVOL-X/b4_delta_reaudit/iiac_audit_session_manifest.EVOL-X.b4_reaudit.json` (passes both validators) has `audited_implementation_sha` `fdb1e7a6215f8eb03bc3ca8d00cce9659dc5c3db`, which is the round-5 audited code pin named in `PUBLICATION_PROVENANCE_NOTE.md` of the W4-01 package; the `b4_delta_reaudit/` directory holds the round-5 dossier. Neither draft's `audited_implementation_sha` (section 3) equals that pin.

No tracked record states which manifest, if any, replaces Draft B or Draft C, or that the certification manifest replaces Draft A. That statement belongs to the EVOL-X certification owner or the Operator and is **not made here**.

## 6. What this record does not do

- It does not edit, move or delete any manifest, evidence index or provenance note, and it changes no validator, allowlist or test.
- It does not decide whether the validators should tolerate listed drafts (a corpus-verifier allowlist is a script change, planned separately) and it does not change the IIAC activation pin or any material governance path.
- It does not state a verdict, a stage status or an independent-verification claim for EVOL-X or any other stage.

## 7. Open point for the owner

Whether to declare, in the owner's records, that the certification manifest replaces Draft A and which manifest (if any) replaces Drafts B and C. Until then the three files stay what the validators show: non-conforming drafts.
