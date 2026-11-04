# ARCH-FREEZE closed-world gap register

**Baseline:** `967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a`

| FINDING_ID | SEVERITY | REQUIREMENT | LOCATION | EVIDENCE | CAUSE | IMPACT | CORRECTION | TEST | OWNER | ACCEPTANCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FIND-BND03-DEBT-STALE | MAJ | FRZ-BND-03 | checklist row + debug_lab + host_observability_composition | @ `967ef2f`: `EvidencePersistencePort`; no Execution→`RuntimeEventPersistence` import | Checklist FRZ-BND-03 evidence stale | FRZ-DEBT blocked until reconciled | T-B2-BND03-RECONCILE | layer boundary gate; ENT-A-BND-* | implementer | Evidence reconciled; status unchanged until promotion |
| FIND-PROD-REACH-DEFERRED | BLOCKER | ECAP §2.C; surface inventory | full production Python tree | Planning: 85 manifest + 8 anchors only | ~7k-file scan not executed | Closed-world audit **not** complete | T-B1-PROD-REACH-SCAN | COMPAT/PROD-Q inventory patterns; ENT FULL | implementer | Extended inventory; UNCLASSIFIED=0 |
| FIND-VBN-PARTIAL | BLOCKER | VBN §2.7; FRZ-RPL/* OPEN | pre-admission §2 VBN-CURRENT | EBH-5 + ENT vbn domain | CURRENT_HEAD audit incomplete | ECAP-§15 fail | T-B2-VBN | ENT VBN 9 checks | implementer | CURRENT_HEAD audit PASS |
| FIND-FORMAL-B6 | BLOCKER | FRZ-EVOL; FRZ-AUD; FRZ-FRZ-11 | ENT_EVOL_CROSS_STAGE_DECISION_MATRIX | TES pins exist; B6 unsigned | External auditor gate | ARCH-FREEZE blocked | T-B5/T-B6 external chain | Protocol 03 replay | independent_auditor | B6_ACCEPTANCE_PUBLICATION_SHA set |
| FIND-ACCEPTED-BASELINE | BLOCKER | FRZ-FRZ-13; ENT FREEZE | ENT_AUDIT_X_SHA_LADDER | ACCEPTED_BASELINE_SHA unset | OD-ENT-01 pending | FREEZE mode blocked | T-B6-OD-ENT-01 | promote CLI | operator | Baseline field populated |
| FIND-PRECHECKLIST-HEAD | MIN | pre-admission dossier | ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md | Still cites 81829712/1b82324 | PIC-AUD-001 @ 967ef2f not reflected | Stale admission read | T-B0-001 doc refresh | L1 link check | implementer | HEAD pointer updated |
| FIND-PATHB-LIVE | INFO | FRZ-PRD-01/02; partial 03/08/SEC-04 | checklist L64 | Path B policy | Dimension B not in ARCH-FREEZE | Must not false PASS live | T-PATHB-EXCL | — | operator | Documented exclusions |
