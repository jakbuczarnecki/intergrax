# Historical stage SHA and evidence registry
Generated 2026-11-08T11:15:30Z from git (`origin/development` = `41908e3fb`). Every SHA below was resolved with `git rev-parse` in the local object store; "UNVERIFIED" means it could not be resolved. A SHA that exists is not evidence that the claim attached to it is true - the claim comes from the cited campaign record.

## 1. SHAs
| group | label | SHA | date | subject | on origin/development |
|---|---|---|---|---|---|
| campaign | CAMPAIGN_START_SHA (origin/development at start) | ba07ebaa34d0 | 2026-11-08 | docs(roadmap): PROD-Q-P0-R1^8 labelled CLOSED - OPERATOR ACCEPTED (scoped) (OD-SSOT-D3, de | True |
| campaign | integration base for the 3 PRs | 15347255ac62 | 2026-11-08 | Merge pull request #404 from jakbuczarnecki/ent-w1-d | True |
| campaign | origin/development after the Operator merged PRs 405-407 | 41908e3fb85d | 2026-11-08 | Merge pull request #405 from jakbuczarnecki/hist-cert/docs-patches | True |
| campaign | PR 405 head (docs patches) | 6d1434796396 | 2026-11-08 | docs: address independent review of the documentation patch set (revert EVOL-X stage-cell  | True |
| campaign | PR 406 head (rules update) | 937c39820885 | 2026-11-08 | docs(iiac): correct two defects found in the delta re-review of the Amendment 01 revision  | True |
| campaign | PR 407 head (T-IGX-030-A) | 90fae38aa39c | 2026-11-08 | docs(tickets): T-IGX-030-A ready_for_audit with implementer checkpoint (no PASS claimed) | True |
| campaign | T-IGX-030-A implementation SHA audited by aud-030a (C2) | 683f955727fb | 2026-11-08 | test(f3): make streamable-HTTP parent-cancel test event-driven (T-IGX-030-A) | True |
| campaign | T-IGX-030-A C1 | ec105a789b8d | 2026-11-08 | test(tool-life-x): reconcile four stale side-effect-safety tests with ABANDONED semantics  | True |
| campaign | T-IGX-030-A plan commit | 78657d445ea3 | 2026-11-08 | docs(tickets): re-plan T-IGX-030 per DR-12=B (030-A tests-only; 030-B and new 030-C HELD f | True |
| campaign | merge commit PR 405 | 41908e3fb85d | 2026-11-08 | Merge pull request #405 from jakbuczarnecki/hist-cert/docs-patches | True |
| campaign | merge commit PR 406 | feee82b2e824 | 2026-11-08 | Merge pull request #406 from jakbuczarnecki/hist-cert/rules-update | True |
| campaign | merge commit PR 407 | 4e48eba82b43 | 2026-11-08 | Merge pull request #407 from jakbuczarnecki/hist-cert/impl-tlx | True |
| historical | QUAL-X Population A accepted SHA | d9ea0d6e87b9 | 2026-10-23 | QUAL-X-FINAL BND-03: commit observability qual repair and SQLite journal E2E. | True |
| historical | QUAL-X publication (per HIST-04 matrix) | b0286f76956c | 2026-11-08 | docs(roadmap): QUAL-X Population A accepted and QUAL-X-P0 verified separately (OD-QUALX-AC | True |
| historical | SKILL-X publication (per HIST-04 matrix) | 756b2d60c5d4 | 2026-11-08 | docs(roadmap): SKILL-X re-acceptance on AUD-SKILLX, conditional (OD-SKILLX-ACCEPT, delegat | True |
| historical | TOOL-LIFE-X accepted baseline (AUD-TOOLLIFE CONDITIONAL / STOP) | 1548640708c8 | 2026-10-26 | fix(qual): F8-Q01 deterministic SQLite oracle fault and Q02 evidence SHA | True |
| historical | SKILL-X Batch C (origin of the P2-C regression, F-015) | d1040f485d7e | 2026-10-23 | SKILL-X Batch C: durable execution skill pinning with provenance recovery. | True |
| historical | TOOL-LIFE-X Batch C recovery code (origin of the EBH-4 R1 import, F-013) | a5f813a2c6a2 | 2026-10-24 | fix(runtime): TOOL-LIFE-X Batch C stalled/abandoned/recovery lifecycle | True |
| historical | TOOL-LIFE-X tool_operation_recovery foundation | ce57be26c63d | 2026-10-24 | TOOL-LIFE-X L3-R6: tenant-safe external operation foundation. | True |
| unpublished | source commit C1 (tests) | 0adb7338075d | 2026-11-08 | test(tool-life-x): reconcile four stale side-effect-safety tests with ABANDONED semantics  | False |
| unpublished | source commit C2 (test) | 10089121e3e1 | 2026-11-08 | test(f3): make streamable-HTTP parent-cancel test event-driven (T-IGX-023-B) | False |
| unpublished | HELD product fix, invoker.py (030-C) | 29c387879a43 | 2026-11-08 | fix(nexus-tools): terminalise external-operation record when pool.submit fails without dep | False |
| unpublished | HELD red-first test (030-C, C4) | 98b4bf514db6 | 2026-11-08 | test(tool-life-x): red-first tests for no-boundary pool.submit failure terminal state (T-I | False |
| iiac | IIAC audit execution SHA (ACT-N11 reference) | ae467c8c4dd9 | 2026-11-05 | Record ACT-01 gate and evidence-preservation SHAs. | True |
| iiac | IIAC policy effective_from / activation phase 1 | f8c29fb8dfe5 | 2026-11-05 | IIAC OD-01 phase 1: publish IN_FORCE activation record (effective SHA pending). | True |
| iiac | Amendment 01 status commit | d82f8b722ca6 | 2026-11-07 | docs(iiac): adopt Amendment 01 (operator-designated signing session) as policy text per op | True |
| evol-x | EVOL-X B6 acceptance publication | 1dd769331506 | 2026-11-07 | docs(evol-x): publish IIAC B6 acceptance record, phase 1 (delegated operator signature; b6 | True |
| evol-x | EVOL-X formal closure publication | be77b3a5fb97 | 2026-11-07 | docs(evol-x): W7-05 formal closure publication, phase 1 (CLOSED - independently accepted,  | True |
| evol-x | EVOL-X W7-06 handoff pin | 7a4f279fd8c0 | 2026-11-08 | docs(evol-x): W7-06 phase 1, formal ENT-AUDIT-X handoff record and D10 digest-chain attest | True |

Notes: SHAs in the group "unpublished" were not on any `origin` branch before PR 407; `0adb73380` and `10089121e` are now published as blob-identical tests via T-IGX-030-A, `29c387879` and `98b4bf514` remain HELD (T-IGX-030-C). "on origin/development = False" for them is expected.

## 2. Stage and lineage entries (HIST-04 supersession matrix, baseline `ba07ebaa3`)
| entry | label at baseline | HIST-04 classification | roadmap line | publication state |
|---|---|---|---|---|
| TRACE-X-P5-R2-P3-R1-R1-R1-R1 | SUPERSEDED - LINEAGE ONLY (OD-SSOT-D1, delegated; commit e7eabef81) | NO_REMAINING_OBLIGATION | L392 | PUBLISHED |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1 | SUPERSEDED - LINEAGE ONLY (OD-SSOT-D1, delegated; commit e7eabef81) | NO_REMAINING_OBLIGATION | L393 | PUBLISHED |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1 | SUPERSEDED - LINEAGE ONLY (OD-SSOT-D1, delegated; commit e7eabef81) | NO_REMAINING_OBLIGATION | L394 | PUBLISHED |
| TRACE-X-P5-R2-P4-R1 | SUPERSEDED - LINEAGE ONLY (OD-SSOT-D1, delegated; commit e7eabef81) | NO_REMAINING_OBLIGATION | L399 | PUBLISHED |
| TRACE-X-P5-R2-P4-R1-R1 | SUPERSEDED - LINEAGE ONLY (OD-SSOT-D1, delegated; commit e7eabef81) | NO_REMAINING_OBLIGATION | L400 | PUBLISHED |
| TRACE-X-P5-R2-P4-R1-R1-R1 | SUPERSEDED - LINEAGE ONLY (OD-SSOT-D1, delegated; commit e7eabef81) | NO_REMAINING_OBLIGATION | L401 | PUBLISHED |
| TRACE-X-P5-R2-P4-R1-R1-R1-R1 | SUPERSEDED - LINEAGE ONLY (OD-SSOT-D1, delegated; commit e7eabef81) | NO_REMAINING_OBLIGATION | L402 | PUBLISHED |
| TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1 | SUPERSEDED - LINEAGE ONLY (OD-SSOT-D1, delegated; commit e7eabef81) | NO_REMAINING_OBLIGATION | L403 | PUBLISHED |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1 | [ ] READY FOR AUDIT | VALID_SCOPED_PASS_CANDIDATE | L395 | PUBLICATION_PENDING (SIGN-6, unpublished) |
| TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1 | [ ] READY FOR AUDIT | VALID_SCOPED_PASS_CANDIDATE | L404 | PUBLICATION_PENDING (SIGN-6, unpublished) |
| PROD-Q-P0 | CLOSED - OPERATOR ACCEPTED / SCOPED PROD-Q-P0 | INDEPENDENT_AUDIT_GAP | L466 | PUBLISHED |
| PROD-Q-P0-R1 .. R1^4 | [ ] REJECTED / superseded (checkbox unchecked) | SSOT_DRIFT | L467-L470 | DECISION_UNPUBLISHED |
| PROD-Q-P0-R1^8 | CLOSED - OPERATOR ACCEPTED (scoped, as implementation evidence of PROD-Q-P0 B6 @ 8b3db676) | INDEPENDENT_AUDIT_GAP | L474 | PUBLISHED (ba07ebaa3) |
| PROD-Q-P2 / P3 / PROD-Q | CLOSED BY OPERATOR - SCOPED (live qualification NOT QUALIFIED) | PATH_B_LIVE_DEFERRED | L475, L476, L478 | PUBLISHED |
| QUAL-X | CLOSED - POPULATION A ONLY (independently accepted, delegated) | ACCEPTED_LIMITATION | L480 | PUBLISHED (b0286f769) |
| QUAL-X-P0 | INDEPENDENTLY VERIFIED - SCOPED (not a stage closure) | ACCEPTED_LIMITATION | L479 | PUBLISHED |
| SKILL-X | CLOSED - INDEPENDENTLY ACCEPTED (conditional on DS-5) | VALID_SCOPED_PASS_CANDIDATE | L481 | PUBLISHED (756b2d60c) |
| TOOL-LIFE-X | CLOSED - INDEPENDENTLY ACCEPTED | SSOT_DRIFT | L482 | PUBLISHED label unchanged |
| DIST-SCALE-X | CLOSED - OPERATOR SCOPED ACCEPTANCE | INDEPENDENT_AUDIT_GAP | L483 | PUBLISHED |
| EVOL-X | stage row: [ ] FORMAL PROTOCOL 03 B6 - PENDING_AUTHORITY (stale) | SSOT_DRIFT | L489 (stale) vs L1565/L561/ladder | PATCH_H4_01 proposed |

## 3. What the campaign changed per stage
| stage | campaign update |
|---|---|
| TOOL-LIFE-X | Conditions C-1/C-2 addressed by T-IGX-030-A (tests only), independently audited PASS_SCOPED at 683f95572; stage NOT closed (EBH-4 R1 gate red because of its recovery code; product batch HELD; label decision DR-3 open). |
| EVOL-X | AUTHORITY_GAP unchanged (delegated plain-form signatures; campaign delegations exclude EVOL-X). Roadmap row L489 prose says CLOSED while its stage cell says PENDING_AUTHORITY (base inconsistency, left for the Operator, DR-1). |
| SKILL-X | Acceptance conditional on DS-5 (met per HIST-04). Batch C d1040f485 introduced the PROD-Q P2-C regression (F-015, worker thesis: code defect) that the acceptance did not catch. |
| QUAL-X | Population A only; global FRZ-REG-01..10 stay OPEN; replay of Dimension A on the final SHA approved (DR-10) but not yet run. |
| PROD-Q | Live Dimension B remains unqualified (Path B); P2-C HTTP admission tests are red at the base (F-015). |

## 4. Evidence locations
- Worker dossiers (per-ID matrices, findings registers, handoffs): `dossiers/hist01..hist04/`.
- Independent reviews and audit: `audits/`.
- Delegation record used for the T-IGX-030-A plan approval: `delegations/DELEG-HIST-030A-2026-11-08.md` (MASTER transcription of the Operator's chat instruction; NOT signed by the Operator; sha256 in the adjacent `.sha256`).
- The `Approved:` line in `governance/tickets/T-IGX-030-A.md` names the delegation by its original local path (`.tmp/session/hist-cert-campaign/delegations/DELEG-HIST-030A-2026-11-08.md`); the published copy is `delegations/DELEG-HIST-030A-2026-11-08.md` (sha256 `e17d1cb131cdbf6475b3330021ed85807979aaaab6292322805528db10a80364`).
- Local evidence that is NOT published and that no claim here depends on exclusively: replay logs of the workers (`hist02/replay_ba07eba.log`, `hist03/pytest_run1.log`, `hist03/iso_*.log`), audit logs (`audits/aud-030a/*.log`).
