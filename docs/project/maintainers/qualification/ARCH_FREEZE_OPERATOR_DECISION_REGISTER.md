# ARCH-FREEZE operator decision register

| OD_ID | PURPOSE | PREREQUISITES | EVIDENCE | ROLE | SIGNATURE | TIMING | CONSEQUENCE | STATUS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
OD-IC-01 | Internal EVOL+ENT acceptance | TES/FINAL/CI | 5c08e5e6… | operator | recorded 2026-11-04 | done | does NOT substitute B6 | COMPLETE |
OD-EVOL-11 | Appoint EVOL auditor | mandate | evol_x/ | operator | required | before EVOL B6 | enables B6 | OPEN |
OD-EVOL-FORMAL | EVOL stage closure publication | EVOL B6 + FRZ-EVOL | W7-06 + roadmap §5 | operator+auditor | required | after FRZ-EVOL; before OD-ENT-00 | EVOL CLOSED | OPEN |
OD-ENT-00 | EVOL dependency for ENT | **T-B5-EVOL-FORMAL-CLOSURE** | ENT_EVOL_CROSS_STAGE | operator | required | after EVOL formal closure | unblocks ENT | OPEN |
OD-ENT-01 | Promote accepted baseline | ENT B6 | ENT ladder | operator | required | post ENT B6 | FREEZE mode | OPEN |
OD-ENT-02 | Branch protection | ENT B6 | OD_ENT_02 packet | github_admin | required | post ENT B6 | CI enforcement | BLOCKED_EXTERNAL |
OD-ENT-08 | ENT stage closure | FRZ-AUD + OD-ENT-01 + **OD-ENT-02** | ECAP | operator | required | pre FREEZE lock | ENT CLOSED | OPEN |
OD-ARCH-FREEZE-01 | ARCH-FREEZE admission | ECAP-§15 | admission dossier | operator | required | post §2.C | freeze authorized | OPEN |
