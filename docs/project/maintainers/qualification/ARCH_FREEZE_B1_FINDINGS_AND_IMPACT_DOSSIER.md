# ARCH-FREEZE B1 — findings and FRZ impact dossier

**EXECUTION_SHA:** `18a0f68fc6c8107e6344fdb4641ecec1945dae5e`

| FINDING_ID | SEVERITY | SOURCE_PATH | REACHABILITY | SEMANTIC_OWNER | INVARIANT | FRZ_ID | IMPACT | REQUIRED_REMEDIATION | TEST | ACCEPTANCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FIND-PROD-REACH-EXECUTED | INFO | arch_freeze_b1_inventory_summary.json | 9024 modules classified | COMPAT-X / PROD-Q | closed-world inventory | FRZ-BND-03,FRZ-BND-04,FRZ-CTR-02 | Supersedes planning deferral | T-B1-PROD-REACH-SCAN **COMPLETE** | arch_freeze_b1_closed_world_inventory.py | mandatory tier UNKNOWN=0 |
| FIND-CW-GATE-FAIL | MAJ | closed_world_gates @ HEAD | CW-05,CW-08,CW-09,CW-10 | COMPAT-X | EVOL CW suite | FRZ-CMP-01,FRZ-CMP-06 | Inventory complete; gates not green @ HEAD | B2: reconcile CW-05 collect manifest; investigate CW-08..10 | test_evol_x_closed_world_gates.py | All CW gates PASS @ certifying SHA |
| FIND-CW-EXTRACTOR-UNKNOWNS | MAJ | closed_world.unknowns | 21 `unknown:` nodes | COMPAT-X | import/graph extraction | FRZ-BND-03,FRZ-CTR-02 | Dynamic/unresolved symbols recorded | B2: behavioral proof or exclusion register update | CW-01..CW-04 | Zero mandatory-scope silent unknowns |
| FIND-BND03-DEBT-STALE | MAJ | debug_lab_nexus_loop.py | LAB surface in strict tier tree | Execution/Nexus | tier boundary | FRZ-BND-03,FRZ-DEBT-03 | Checklist debt text stale | T-B2-BND03-RECONCILE | test_hardening_3_layer_boundary_gate.py | Evidence reconciled |
| FIND-DEBT-REG | MAJ | debt registers | architecture OPEN debt | Governance | FRZ-DEBT family | FRZ-DEBT-01..06 | Blocks FRZ-DEBT promotion | T-B2-DEBT | ENT debt checks | Zero unresolved debt |
| FIND-VBN-PARTIAL | BLOCKER | VBN domain | vendor replaceability | ENT VBN | FRZ-RPL | FRZ-RPL-01..04 | ECAP-§15 risk | T-B2-VBN | ENT VBN 9 checks | CURRENT_HEAD PASS |
| FIND-FORMAL-B6 | BLOCKER | EVOL/ENT stages | external gate | independent_auditor | Protocol 03 | FRZ-EVOL,FRZ-AUD,FRZ-FRZ-11 | Blocks freeze admission | T-B5/T-B6 B6 | TES replay | B6 SHA set |
| FIND-ACCEPTED-BASELINE | BLOCKER | ENT ladder | baseline unset | operator | FREEZE baseline | FRZ-FRZ-13 | Blocks FREEZE mode | T-B6-OD-ENT-01 | promote CLI | ACCEPTED_BASELINE_SHA |
| FIND-PATHB-LIVE | INFO | checklist L64 | live PRD/SEC slices | operator | Dimension A/B | FRZ-PRD-01/02; FRZ-SEC-04 | No implicit global PASS | T-PATHB-EXCL | Path B table | Documented exclusions |

## FRZ impact summary (no status promotion)

- **FRZ-BND / FRZ-CTR:** inventory surfaces validated; BND-03 evidence reconciliation still **OPEN** (T-B2).
- **FRZ-PRD:** Path B split enforced in WBS (`T-B7-PROMOTE-PRD` = FRZ-PRD-03..12 only; 01/02 → `T-PATHB-EXCL`).
- **FRZ-CMP:** CW gate failures @ HEAD require B2 qualification attention before freeze certification.
- **Remaining OPEN families:** unchanged checklist disposition; inventory enables B2 batching only.
