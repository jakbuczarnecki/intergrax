# ARCH-FREEZE closed-world gap register

**Baseline:** `a0624e28ff77b061e83f3e10aab4c2a9083ee1b4`

| FINDING_ID | SEVERITY | REQUIREMENT | LOCATION | EVIDENCE | CAUSE | IMPACT | CORRECTION | TEST | OWNER | ACCEPTANCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FIND-PRECHECKLIST-HEAD | MIN | pre-admission dossier | ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md | Stale HEAD pointer | PIC-AUD-001 not reflected @ a0624e28 | Stale admission read | T-B0-SSOT-REFRESH | L1 link check | implementer | HEAD pointer updated |
| FIND-BASELINE-SHA-DRIFT | MAJ | FRZ-FRZ-04 | ARCH_FREEZE_* dossier | SSOT locked 967ef2f vs HEAD a0624e28 | Planning lag | Wrong replay baseline | T-B0-SSOT-REFRESH | verify_arch_freeze_plan_traceability.py | implementer | AUDIT_BASELINE_SHA=a0624e28 |
| FIND-ARTIFACT-CONTRADICTION | MAJ | governance | traceability vs WBS status | NOT EXECUTED vs COMPLETE | Wave-1 partial sync | Operator confusion | T-B0-SSOT-REFRESH | L1-TRACE | implementer | No contradictory scan status |
| FIND-VALIDATOR-WEAK | MAJ | mechanical proof | verify_arch_freeze_plan_traceability.py | Hardcoded WBS=39 | Incomplete validator | False PASS | T-B0-SSOT-REFRESH | validator exit 0 | implementer | Dynamic WBS=48 checks |
| FIND-PROD-REACH-CLASSIFICATION | BLOCKER | ECAP §2.C | arch_freeze_b1_closed_world_inventory.py | Overbroad PRODUCTION_EFFECTIVE | Path-default classification | False production surface | T-B1-ENGINE | test_arch_freeze_b1_closed_world_inventory.py | implementer | PRODUCTION_EFFECTIVE only w/ strict reachability |
| FIND-UNKNOWN-HARDCODE | MAJ | B1 acceptance | inventory summary | unknown_modules=0 hardcoded | Silent unknowns | B1 false PASS | T-B1-ENGINE | B1 unit tests | implementer | Honest unknown_modules |
| FIND-CW-GATE-FAIL | MAJ | FRZ-CMP-01..08 | closed_world_gates | CW-05/08/09/10 fail without inputs | Inventory invoked gates minimally | Blocks B1 strict | T-B1-ENGINE; T-B1-STRICT-QUALIFY | test_evol_x_closed_world_gates.py | implementer | All CW gates PASS @ certifying SHA |
| FIND-CW-EXTRACTOR-UNKNOWNS | MAJ | FRZ-BND-03,FRZ-CTR-02 | closed_world unknowns (21 nodes) | Extractor limits | Static graph incomplete | B1 disposition required | T-B1-MANDATORY-DISPOSITION | CW-01..CW-04 | implementer | All mandatory unknowns dispositioned |
| FIND-CW-UNRESOLVED-IMPORTS | MAJ | FRZ-BND-03,FRZ-CTR-02 | closed_world.unresolved_imports | Mandatory-scope imports | Static graph incomplete | B1 disposition required | T-B1-MANDATORY-DISPOSITION | strict inventory | implementer | Zero undisposed mandatory imports |
| FIND-B1-EXIT-CODE | MAJ | qualification | inventory CLI | exit 0 on gate fail | No strict mode | False PASS | T-B1-ENGINE | B1 unit tests | implementer | --strict-certify exit 1 on fail |
| FIND-B1-SHA-PAYLOAD | MAJ | exact-SHA ladder | inventory JSON | audit_baseline hardcoded | Stale field | Wrong evidence lineage | T-B1-ENGINE | summary JSON schema | implementer | implementation + baseline SHA fields |
| FIND-BND03-DEBT-STALE | MAJ | FRZ-BND-03 | checklist + debug_lab | EvidencePersistencePort @ HEAD | Checklist debt text stale | FRZ-DEBT blocked | T-B2-BND03-RECONCILE | test_hardening_3_layer_boundary_gate.py | implementer | Evidence reconciled |
| FIND-DEBT-REG | MAJ | FRZ-DEBT-01..06 | debt registers | OPEN FRZ-DEBT family | Unresolved debt | Blocks promotion | T-B2-DEBT; T-B7-PROMOTE-DEBT | ENT debt checks | implementer | Zero blocking debt |
| FIND-VBN-PARTIAL | BLOCKER | FRZ-RPL-01..04 | VBN-CURRENT | Partial taxonomy | CURRENT_HEAD gaps | ECAP-§15 risk | T-B2-VBN; T-B7-PROMOTE-RPL | ENT VBN | implementer | VBN PASS @ HEAD |
| FIND-FORMAL-B6 | BLOCKER | FRZ-EVOL; FRZ-AUD; FRZ-FRZ-11 | cross-stage matrix | B6 unsigned | External gate | ARCH-FREEZE blocked | T-B5-EVOL-B6; T-B6-ENT-B6 | Protocol 03 replay | independent_auditor | B6 SHA set |
| FIND-ACCEPTED-BASELINE | BLOCKER | FRZ-FRZ-13 | ENT ladder | ACCEPTED_BASELINE unset | OD-ENT-01 pending | FREEZE blocked | T-B6-OD-ENT-01 | promote CLI | operator | Baseline populated |
| FIND-PATHB-LIVE | INFO | FRZ-PRD-01/02 | checklist L64 | Path B policy | Dimension B deferred | No false global PASS | T-PATHB-EXCL | — | operator | Exclusions recorded |
| FIND-PROD-REACH-DEFERRED | INFO | ECAP §2.C | wave-1 inventory @ 18a0f68 | Historical scan | Superseded by B1 chain | — | T-B1-PROD-REACH-SCAN | arch_freeze_b1_closed_world_inventory.py | implementer | Historical only |
