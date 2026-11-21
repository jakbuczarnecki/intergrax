<!-- Draft added by governance/tickets/T-IGX-042-D.md (EQO-A remediation, documentation only). Not adopted until that ticket carries an Operator approval line. -->

# Enterprise Debt & Exceptions Register (EDER)

Status: **DRAFT - not adopted.** Authority: Operator decision OD-REM-04 / condition C4 (EQO-A remediation planning, 2026-11-16); location per Operator decision DP-46 (same date). This document describes the register; it does not certify, promote or close anything.

## 1. Purpose

EDER is the single place where every known **debt item, evidence gap, pending decision and accepted exception** of the Enterprise program can be found, with one accountable owner, one severity, one ARCH-FREEZE impact class, one exact-SHA source, one disposition and one closure condition. It replaces the question "which of the 28 scattered registers is current?" with one traceable list. It does not replace those registers.

## 2. What EDER is not (non-duplication rule)

1. **EDER owns tracking metadata only.** Canonical facts stay where they are: stage status in the roadmap, FRZ criterion status in the freeze checklist, domain ownership in the domain architecture documents and `PDC_OWNERSHIP_MATRIX`, decisions in their signed decision records, certification text in the stage records.
2. A row points to its source by `path@sha:line`, to its owner by ID, to related canonical objects by ID. It never copies their prose, never restates a stage or FRZ status, and its `title` is a label (max 600 characters).
3. **The canon moves first.** A row may only be closed after the canonical source it points to has been updated (same pull request is allowed). If EDER and a canonical source disagree, the canonical source wins and the row is re-opened by the consistency check.
4. EDER never promotes a FRZ status, never claims PASS or certification and never records "independently verified" without a preserved independent audit artefact (reference in `independent_audit_ref`).
5. A new ownership claim needs a decision record. A row may *propose* an `owner_ref`; it becomes `CONFIRMED` only through the confirmation step in section 6.

## 3. Identity

`EDER-<K>-<NNNN>`: `F` = ingested EQO-A finding, `D` = ingested register/ledger entry, `X` = exception recorded without a prior row, `N` = raised after the initial ingest. IDs are immutable and never reused. Merged rows keep their ID with state `SUPERSEDED` and `superseded_by`. Source identity is `origin.id`: the finding ID or the dossier entry `uid` (the dossier `entry_id` is not unique: 48 values repeat) and must be unique among non-superseded rows.

## 4. Row fields (summary; the machine schema is `register_schema.json`)

| Group | Fields |
|---|---|
| Identity | `reg_id`, `record_type`, `register_class`, `origin{kind,id,entry_id,origin_register}`, `title` |
| Ownership | `owner_ref` (ID), `co_owner_refs`, `owner_status`, `owner_raw`, `owner_stage_status`, `accountable_person` (Operator-assigned only) |
| Risk | `severity` (`BLOCKER`,`HIGH`,`MEDIUM`,`LOW`,`INFO`,`UNRATED`), `severity_source`, `triage_priority` (ordering aid, not a rating) |
| Evidence | `source_evidence[]` (`path@sha40:line`), `source_evidence_narrative[]`, `baseline_sha`, `snapshot_shas[]`, `current_head_sha`, `current_head_status`, `current_head_evidence[]` |
| Freeze impact | `frz_arch_freeze_impact` (`BLOCKS`,`CONDITIONS`,`NONE`,`UNKNOWN`), `frz_impact_basis`, `frz_refs[]` |
| Handling | `disposition`, `next_track`, `secondary_tracks[]`, `closure_condition`, `reopen_condition`, `needs_independent_audit` (`yes`,`no`,`tbd`) + reason |
| Lifecycle | `lifecycle_state`, closure/acceptance fields (below), `stale_flag`, `linked_finding_ids[]`, `canonical_refs[]` |

Meaning of the impact classes: **BLOCKS** - the ARCH-FREEZE entry condition cannot be met while the row is open. **CONDITIONS** - admission is possible only with the stated condition or an accepted limitation. **NONE** - no effect, with the basis written down. **UNKNOWN** - not yet classified; a row in this class may not be deferred and may not become an accepted exception.

`register_class` values: `DEFECT`, `ARCH_RISK`, `GOVERNANCE_DEBT`, `EVIDENCE_GAP`, `CERT_DRIFT`, `UNVERIFIED`, `EXCEPTION_CANDIDATE`, `DEBT`, `DECISION_PENDING`, `KNOWN_RED_TEST`, `NA_DECISION`, `SKIP_CATEGORY`. The last three are defined in section 9.

`owner_ref` is one program- or stage-level ID per row (optional `co_owner_refs`), taken from an owner vocabulary of IDs that the Operator confirms once (Operator decision DP-47). The vocabulary file and the register rows that name owners are not part of this change; rows follow after the Operator has confirmed the vocabulary. Until then no row may be moved past `INGESTED`.

`current_head_status`: `UNCHANGED` means the cited tracked paths are byte-identical at the current head; it does **not** mean the claim was re-verified. `RESOLVED` is only valid together with closure evidence. `UNKNOWN` means no SHA-pinned tracked path exists to compare.

## 5. Lifecycle

`INGESTED -> TRIAGED -> PLANNED -> IN_PROGRESS -> VERIFYING -> CLOSED`; side exits `EXCEPTION_ACCEPTED`, `SUPERSEDED`, `WITHDRAWN`; `REOPENED -> TRIAGED`.

| State | Entry requirement (checked mechanically) |
|---|---|
| INGESTED | Source resolves; baseline SHA; all structural fields present. Owner, severity, impact class and track may be provisional. |
| TRIAGED | `owner_status=CONFIRMED`; `severity != UNRATED`; impact class != `UNKNOWN`; `next_track != UNCLASSIFIED`; closure and reopen conditions authored for this row (not template); `needs_independent_audit` decided. |
| PLANNED | Ticket, decision request or plan reference recorded in `canonical_refs`. |
| IN_PROGRESS | Work started under that reference. |
| VERIFYING | Evidence submitted at an exact SHA. |
| CLOSED | `closure_evidence`, `closed_at_sha`, `closed_by`, `reviewed_by` (a different party); `independent_audit_ref` when `needs_independent_audit=yes`; stale rows additionally `reverified_at_sha`. |
| EXCEPTION_ACCEPTED | Acceptance record (`path@sha`), accepting authority, scope reference, impact class other than `UNKNOWN`, review date or reopen condition. |
| SUPERSEDED / WITHDRAWN | Reference to the surviving row, or evidence at an exact SHA that the item is not a debt. |

## 6. Who may change what

| Action | Allowed | Not allowed |
|---|---|---|
| Create `INGESTED` rows, edit labels and links | Register steward under a docs ticket | Anyone without a ticket |
| Propose owner / severity / impact class | The named owner, the steward | The implementer of the fix for its own row |
| Confirm `owner_ref` | The Operator, or the owner program named in the roadmap with Operator-assigned `accountable_person` | The steward alone |
| Set impact `BLOCKS` or change it | ARCH-FREEZE owner, confirmed by the Operator | The row owner alone |
| Move to `VERIFYING` | Implementer / owner | - |
| Move to `CLOSED` | A reviewer who is not the implementer; an independent G6 auditor session where `needs_independent_audit=yes` | The implementer. No self-PASS |
| `EXCEPTION_ACCEPTED` | The Operator, or a delegate under the documented delegation procedure (IIAC Amendment 01) | Implementer, owner, steward |
| `REOPENED` | The consistency check or anyone, recorded by the steward | - |

## 7. Maintenance

- **Cadence:** the steward runs the consistency check on every pull request that touches a register source or a row, and at each freeze-relevant milestone.
- **Triggers:** a cited evidence path changes; a stage or FRZ status changes in the canon; a new register or ledger appears (it must be listed in the qualification index and ingested or explicitly excluded); a stale-snapshot register is regenerated.
- **Snapshot registers:** every register that EDER ingests carries one of `LIVE` (maintained by its owner), `SNAPSHOT_AS_OF_<sha>` (historical, rows re-verified before closure) or `SUPERSEDED_BY_EDER`. No source register may remain unlabelled.
- **Reconciliation:** each ingest or refresh publishes counts per source and per exclusion class (see `reconciliation.json`); a refresh that changes totals without a matching source change is rejected.

## 8. Initial content (draft numbers from the Campaign A dossier)

The reconciliation (counts per source and per exclusion class) is to be published next to the register as `reconciliation.json`; that file is not part of this change. The scope of the initial load follows the Operator decisions of 2026-11-16 (DP-50, DP-51):

- 239 active EQO-A findings, 605 A7 entries with currency OPEN or UNKNOWN and the 26 DEFERRED entries are ingested (DP-51).
- 155 count/summary rows carry currency OPEN or UNKNOWN but are not entries; they are excluded and listed separately, so that 605 + 155 = 760 reconciles (DP-51).
- Accepted limitations (456 entries): the 155 accepted A2 and register entries become `EXCEPTION` rows; the 301 IIAC manifest limitations are represented by one pointer row per manifest, linking by path (DP-50).
- Resolved, closed and superseded entries (211) are not ingested in the initial load.

All counts are the dossier's, at its baseline; the ingest re-derives them and publishes the result.

The debt list named "P9 debt register" in the dossier (EQOA-A7-012, -014, -015) is this register (Operator decision DP-52). The matching reference change in the PDC records (PDC-P9-01) is a separate amendment agreed with the PDC program and is not made by this change.


## 9. Standing categories and entries (EQO-A remediation draft, as of 2026-11-16)

These categories exist in the schema (`register_class`, `disposition`). They define how a row of that kind is written; rows are created at ingest, not in this document.

### 9.1 Role: certification-debt register (EQOA-A0-003)

EDER is the certification-debt register of the Enterprise program. Before it, certification limitations and scoped acceptances were distributed over records and ledger notes. It consists of this README, `register_schema.json` and the register document (`register.jsonl`, created at ingest and not part of this change).

### 9.2 Known-red tests (`KNOWN_RED_TEST`, EQOA-A0-001)

The category exists. A row is created only from the output of an authorised bounded run, with the run log cited as `path@sha` source evidence. A static inventory is not a substitute. No `KNOWN_RED_TEST` row exists until such a run has been authorised and cited.

### 9.3 Not-applicable decisions and exceptions (`NA_DECISION`, EQOA-A0-002)

The category holds N/A and exception decisions that today sit inline in qualification records, where their validity cannot be checked mechanically. A row points to the decision (`path@sha:line`), states the condition under which the N/A stays valid, and carries an owner and a reopen condition. At the baseline the dossier counts 558 N/A decision mentions in 147 files; the only N/A register that exists is the EVOL-X deployment N/A register.

### 9.4 Skip, xfail and `no_ci` markers (`SKIP_CATEGORY`, EQOA-A7-037)

EDER holds category rows only: environment-gated, optional dependency, `no_ci`, N/A decision. Per-marker disposition lives in an extended EBH-7 / QUAL-X skip register (today `ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`); EDER does not hold one row per marker. At the baseline a static scan of `tests/` finds 691 skip-class markers (`pytest.mark.no_ci` 284, `pytest.skip(` 259, `pytest.mark.skipif` 71, `importorskip` 69, `pytest.mark.skip` 6, `pytest.mark.xfail` 2). The dossier reports 164 high-surface markers outside the EBH-7 skip disposition register files and 13 high-surface `no_ci` markers.

### 9.5 Severity and stale snapshots (EQOA-A7-034, EQOA-A7-035)

`UNRATED` is a valid severity. The register does not infer severities; owners rate rows at triage (Operator decision DP-02: triage by owners, no mechanical ratings; `triage_priority` is an ordering aid only). The dossier counts 405 OPEN or DEFERRED entries without a severity (EQOA-A7-034). A row whose source register is a snapshot that a source at the baseline contradicts carries the disposition `STALE_SNAPSHOT_REVERIFY` and is re-verified before closure; the dossier counts 122 such entries (EQOA-A7-035). The source-register header labels of section 7 are applied by the register owners after the register data exists.

### 9.6 Standing entry: superseded IIAC manifest drafts (EQOA-A2-006)

Three draft IIAC audit-session manifests remain in the evidence tree next to valid ones and carry the placeholder value `PENDING_CERTIFIER` (three occurrences each):

- `docs/project/maintainers/qualification/iiac_evidence/EVOL-X/audit/iiac_audit_session_manifest.EVOL-X.auditor_draft.json`
- `docs/project/maintainers/qualification/iiac_evidence/EVOL-X/b4_reaudit/iiac_audit_session_manifest.EVOL-X-B4.reaudit_draft.json`
- `docs/project/maintainers/qualification/iiac_evidence/EVOL-X-W4-01-REAUDIT/iiac_audit_session_manifest.EVOL-X-W4-01.reaudit_draft.json`

The register is to record them as superseded drafts. This change does not move or edit the files; other records cite their paths. Which certified manifest supersedes each draft is not stated here. Teaching tools to treat these three files as known drafts (a corpus-verifier allowlist) is a script change and is outside this document.

A point-in-time record with the per-draft facts, the validator output, the citations that must keep resolving and the later manifests of the same evidence line is [`ent_audit_x/ENT_AUDIT_X_IIAC_MANIFEST_DRAFT_POINT_IN_TIME_RECORD.md`](../ent_audit_x/ENT_AUDIT_X_IIAC_MANIFEST_DRAFT_POINT_IN_TIME_RECORD.md) (ticket `T-IGX-044-A`). It declares no supersession.

## 10. Owner vocabulary confirmation (note, 2026-11-17)

On 2026-11-17 the Operator's chat decision on DP-47, option a (recorded as addendum 4, item "DP-47 owner model and vocabulary", in [`../eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md`](../eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md); an orchestrator transcription, not an Operator signature) confirmed the 55-ID vocabulary in `owner_vocabulary.proposed.json` as canonical: one program- or stage-level accountable owner per row, optional co-owners, named persons only where the Operator assigns them.

- The file keeps its name (no schema or script reads the name, and other records cite the path); its `status` now reads `CONFIRMED`.
- `owner_vocabulary_status` of every register row now reads `CONFIRMED` (1069 of 1069 rows; it read `PROPOSED_UNCONFIRMED`). `reconciliation.json` carries the new status and the count.
- Not changed: the row-level `owner_status` (no row is `CONFIRMED` at row level), `accountable_person` (empty on every row; no Operator assignment of a person is published), the lifecycle state (every row is `INGESTED`), severities, dispositions and impact classes. No row is closed, accepted as an exception or independently verified.
- Findings `EQOA-A7-023` and `EQOA-A7-033` (rows without a named owner) are not handled by this change: they need per-row owner assignments that follow the DP-34 classification (DP-48) and, for the 14 `OWNER_MISSING` entries, assignments by the Operator.
- The sentence in section 4 that the vocabulary file and the owner-naming rows are not part of this change describes the earlier, structure-only change.
