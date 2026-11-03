# ENT-AUDIT-X — post-implementation final closure master plan

**Disposition:** `ENT-AUDIT-X — FINAL CLOSURE EXECUTION SSOT PUBLISHED / READY FOR AGENT WAVE`

**Mode:** PLAN SSOT (implementation wave follows this document; no B6 PASS / FRZ CLOSED / ARCH-FREEZE declared here)

**CURRENT HEAD / audited implementation baseline:** `34e03425b2322f415a99a7cfd290fe0982bf6970`  
**Historical IMPLEMENTATION_SHA (prior closure wave):** `1fa5347a56c569fa62ac2c777ff3947d11e906b4`  
**Branch:** `development` · **Repo:** `jakbuczarnecki/intergrax`

**PLAN_PUBLICATION_SHA:** `f7243601dbeec93bf26d258c05f545bb1704c7eb`

**R0 closure commit @ HEAD:** `34e03425b2322f415a99a7cfd290fe0982bf6970` — 13 files (OD verify, dry-run, evidence eligibility, readiness fail-closed, qual tests, master plan reconciliation).

**Supersedes (execution):** [ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md](./ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md) — immutable TaskID text for C001–C064.

**Reconciles (R5 doc lifecycle):** [ANNEX_F_GAP_REGISTER.md](./ANNEX_F_GAP_REGISTER.md), [ANNEX_H_WBS.md](./ANNEX_H_WBS.md), [ANNEX_G2_CONTROL_GATE_MAPPING.md](./ANNEX_G2_CONTROL_GATE_MAPPING.md), [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md), [ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md](./ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md), [ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md](./ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md).

### PL-01..PL-12 resolution index

| PL | Topic | Resolution |
| --- | --- | --- |
| PL-01 | GAP counts | **33** GAP-RC (mechanical @ HEAD) + **6** GAP-B + **85** GAP-COV — prior **32** claim omitted GAP-RC-CI-PROFILE in summary only |
| PL-02 | TaskID canon | C001–C064 frozen; C055=Annex F; C057=OD matrix **doc by agent**; C058=B6 handoff tree |
| PL-03 | Atomic WBS | **75** tasks; Matrix C dual-axis; impl {'PARTIAL': 21, 'IMPLEMENTED': 4, 'NOT_STARTED': 50}; qual {'NOT_QUALIFIED': 73, 'QUALIFIED': 2} |
| PL-04 | Qual timing | §7 fifteen-step sequence after R0–R4 integration |
| PL-05 | DAG | §14 topological order + shared-file matrix |
| PL-06 | 85 matrix | Matrix E (authoritative CheckID set) |
| PL-07 | Pytest taxonomy | **184** nodeids @ `34e0342` collect-only |
| PL-08 | OD-ENT-01 | ECAP order: B6 → EVIDENCE_PUBLICATION → OD-ENT-01 → ACCEPTED |
| PL-09 | Readiness | C066; `formally_certified` always false in agent script |
| PL-10 | Dependencies | §14 DAG |
| PL-11 | SSOT lifecycle | This file = single execution SSOT |
| PL-12 | Mechanical READY | §16 checklist PASS @ plan publication |

### TaskID immutability control (PL-02)

- **Registry:** C001–C064 = corrective plan `ENT-T-C001..C064`; C065–C075 = Appendix B in corrective + Matrix C.
- **CI guard:** `tests/qualification/ent_audit_x/unit/test_task_id_registry.py` @ `34e0342`.

---

## 1. Executive audit disposition @ `34e0342`

Tier-0 enterprise audit **exists**; R0 trust wave **partially implemented**; **0/85** semantic certifying PASS @ `TEST_EXECUTION_SHA` for current HEAD (historical ladder TES **`0cee41a88274a53d14bb3558752d55bdde0d0b01`** is **not** current 85/85 proof).

| Authority | Status @ HEAD (read-only) |
| --- | --- |
| ENT-AUDIT-X (roadmap §3.1) | OPEN |
| FRZ-AUD-01..10 | OPEN ([checklist](../../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md)) |
| ARCH-FREEZE | OPEN |
| EVOL B6_ACCEPTANCE_PUBLICATION_SHA | EXTERNAL_PENDING |
| TEST_EXECUTION_SHA (ladder) | Stale until §7 step 3 after R0–R4 |

### 1.1 R0 commit inventory (`34e03425b`)

| Path | TaskIDs |
| --- | --- |
| `intergrax/platform/enterprise_audit/baseline.py` | C065, C003 |
| `intergrax/platform/enterprise_audit/evidence.py` | C068, C006 |
| `intergrax/platform/enterprise_audit/modes.py` | C067, C071 |
| `intergrax/platform/enterprise_audit/runner.py` | C067 |
| `scripts/enterprise_audit/mechanical_readiness_check.py` | C066, C054 |
| `tests/qualification/ent_audit_x/unit/test_*` | C042, C066, C075 |
| This document | C075 |

---

## Matrix A — Roadmap §3.1 + FRZ-AUD-01..10

### A.1 Roadmap §3.1 families

| Requirement | Invariant / checks | Implementation @ HEAD | Current proof | Gap | Task | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| RM-ARCH-BND | Architecture / boundaries | ENT-A-BND-* | runner+manifest | wiring tests | GAP-RC-SEMANTIC-GATES | C034 | 85 gate exec @ TES |
| RM-OWN | Ownership | ENT-A-OWN-* | manifest | entrypoint_exists | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-CTR | Contracts / abstraction | ENT-A-CTR-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-TYP | Strong typing | ENT-A-TYP-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-DUP | Duplication | ENT-A-DUP-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-PLG | Pluginability | ENT-A-PLG-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-VBN | Vendor/backend neutrality | VBN-* + ENT-A-VBN-* | vbn.py | partial script | GAP-RC-VBN-31 | C008-C036 | 31 domains + 9 checks |
| RM-GOV | Governance | ENT-A-GOV-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-EXE | Execution | ENT-A-EXE-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-OBS | Observability | ENT-A-OBS-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-STA | State / persistence | ENT-A-STA-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-SEC | Security / data lifecycle | ENT-A-SEC-* | manifest | wiring | GAP-RC-REDACTION | C015,C034 | redaction+gate |
| RM-REL | Resource / overload | ENT-A-REL-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-REC | Durable recovery | ENT-A-REC-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-PRD | Production operability | ENT-A-PRD-* | report limitations | partial flag | GAP-RC-SEMANTIC-GATES | C034 | scoped PASS |
| RM-CMP | Compatibility | ENT-A-CMP-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-REG | Qualification / debt | ENT-A-REG-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-DOC | Documentation drift | ENT-A-DOC-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-TEN | Tenant isolation | TEN-* (13) | tenant enrich | wiring | GAP-COV×13 | C035 | TEN13 @ TES |
| RM-EVOL | Enterprise evolvability | ENT-A-EVOL-* | changed_since partial | no EVOL bind | GAP-RC-CHANGED-IMPACT | C072,C029 | impact map |
| RM-INV | Platform invariants | ENT-A-INV-* | manifest | wiring | GAP-RC-SEMANTIC-GATES | C034 | gate PASS |
| RM-MOD | Modular runner framework | EnterpriseAuditRunner | package exists | import+CLI | GAP-RC-CERTIFY-SPLIT closed | C004 | FRZ-AUD-01 partial |
| RM-BASE | Baseline / drift | baseline.py | FSM partial | unit tests | GAP-RC-OD-TRUST | C001-C003,C065 | promote ECAP order |
| RM-MODE | FAST/FULL/FREEZE/CHANGED | modes.py+runner | modes exist | dry-run fixed INQ | GAP-RC-DRY-RUN | C067,C009 | L3 modes @ TES |
| RM-RPT | Reports JSON+MD | report.py | reports exist | schema shallow | GAP-RC-JSON-SCHEMA | C013,C069 | schema valid |
| RM-EXIT | Exit codes 0-3 | modes.py | enum exists | L3 loose | GAP-RC-L3-FULL | C004,C067 | strict qual |
| RM-POST | Post-freeze policy | runbook+CI | draft | BLK-CI | GAP-RC-PR-PATH-SKIP | C047,C064 | enforceable policy |

### A.2 FRZ-AUD-01..10

| FRZ | Requirement | Invariant | Implementation | Current proof | Gap | Task | Acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FRZ-AUD-01 | Runner exists | RM-MOD | `EnterpriseAuditRunner` | import + CLI | partial qual | C004,C031 | Modes execute |
| FRZ-AUD-02 | Modular checks | RM-MOD | manifest 85 | wiring | GAP-RC-SEMANTIC-GATES | C019,C030,C033 | 85 gates @ TES |
| FRZ-AUD-03 | FRZ map | RM-REG | digests coarse | partial | GAP-RC-FRZ-MAP | C020,C021 | frz_coverage |
| FRZ-AUD-04 | Reports | RM-RPT | JSON+MD | shallow schema | GAP-RC-JSON-SCHEMA | C013,C062,C069 | jsonschema |
| FRZ-AUD-05 | Exact SHA | RM-RPT | manifest fields | partial | GAP-RC-SHA-LADDER | C048,C059 | SHA = git HEAD |
| FRZ-AUD-06 | Baseline drift | RM-BASE | baseline+drift | unit | GAP-RC-OD-TRUST | C001,C022 | drift vs accepted |
| FRZ-AUD-07 | Drift classes | RM-BASE | REGRESSED/RESOLVED | partial | GAP-RC-RESOLVED-BLOCK | C071,C023 | RESOLVED non-blocking |
| FRZ-AUD-08 | Exit codes | RM-EXIT | modes.py | C067 tests | GAP-RC-L3-FULL | C004,C067 | strict L3 |
| FRZ-AUD-09 | Modes | RM-MODE | 4 modes | CI FAST real | GAP-RC-L3-FULL | C009,C031,C032 | L3 FULL/FREEZE |
| FRZ-AUD-10 | Post-freeze policy | RM-POST | runbook | no enforce proof | BLK-CI-ENFORCE | C047,C064 | operator CI |

---

## Matrix B — AUD-01..44

| AUD-ID | Status | Path / function | Evidence @ HEAD | Root cause | GAP-RC | TaskID | Positive test | Negative test | Closure criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUD-01 | PARTIAL | baseline.py::promote_accepted_baseline L258 | optional bundle verify | GAP-RC-OD-TRUST | C065,C042 | promote w/ bundle required | promote without bundle | Promotion requires closed bundle |
| AUD-02 | CONFIRMED_OPEN | ENT_AUDIT_X_SHA_LADDER.md | TES still 0cee41a8 not HEAD | GAP-RC-SHA-LADDER | C048,C075 | append IMPLEMENTATION row | rewrite history | Ladder matches HEAD lineage |
| AUD-03 | PARTIAL | baseline.py L203-208 | operator_records path enforced | GAP-RC-OD-TRUST | C065 | promote path test | record outside dir | Only operator_records authorize |
| AUD-04 | PARTIAL | evidence.py L51-56 | execution_manifest optional | GAP-RC-EVIDENCE-TRUST | C068 | bundle w/ manifest | None manifest | Manifest required certifying |
| AUD-05 | CONFIRMED_OPEN | l3/test_l3_clean_checkout_full.py | dry-run/single-check; loose exit | GAP-RC-L3-FULL | C031,C074 | strict 85/85 exit 0 | dry-run qual | Real FULL @ TES |
| AUD-06 | PARTIAL | vbn.py::load_vbn_domain_report | script exists; validator C008 open | GAP-RC-VBN-31 | C008-C011 | 31 accounted qual | stale cache | domains_accounted==31 @ TES |
| AUD-07 | FIXED_NOT_QUALIFIED | mechanical_readiness_check.py | fail-closed JSON not is_file | GAP-RC-MECH-READY | C066 | all checks true post-qual | missing L3 | TECHNICAL_READY only w/ proof |
| AUD-08 | CONFIRMED_OPEN | — | 0/85 QUALIFIED @ TES | GAP-RC-L3-FULL | C031-C036,C074 | 85/85 manifest | collect-only | Semantic PASS @ pinned SHA |
| AUD-09 | PARTIAL | changed_since.py | path-only selection | GAP-RC-CHANGED-IMPACT | C024,C072 | impact map selects | unmapped BLOCKER | SSOT impact map |
| AUD-10 | OPEN | EVOL artifacts | not bound CHANGED-SINCE | GAP-RC-CHANGED-IMPACT | C029,C072 | EVOL map reuse | bypass impact | Impact includes EVOL SSOT |
| AUD-11 | OPEN | baseline.py drift | schema_version per-check incomplete | GAP-RC-DRIFT-SCHEMA | C022 | schema bump REGRESSED | stable pass | Per-check schema in manifest |
| AUD-12 | PARTIAL | baseline.py apply_drift | fingerprint identity partial | GAP-RC-DRIFT-SCHEMA | C022,C023 | REGRESSED severity | wrong identity | Stable finding IDs |
| AUD-13 | CONFIRMED_OPEN | validators.py L17-37 | shallow type check only | GAP-RC-JSON-SCHEMA | C069 | jsonschema pass | extra props | Nested schema all artifacts |
| AUD-14 | PARTIAL | sandbox.py + unit test | traversal test exists | GAP-RC-EVIDENCE-TRUST | C027 | symlink escape blocked | path escape | Sandbox jail proven |
| AUD-15 | OPEN | evidence_artifact_catalog.v1.json | no mode profiles | GAP-RC-EVIDENCE-TRUST | C073 | profile match | missing artifact | FAST/FULL/FREEZE profiles |
| AUD-16 | PARTIAL | report.py | per-file write | GAP-RC-ATOMIC-PUB | C013 | interrupt atomic | torn report | All-or-nothing set |
| AUD-17 | OPEN | sandbox.py | isolation/timeouts incomplete | GAP-RC-SANDBOX | C014-C017 | concurrent runs | collision | Isolated runs + kill orphans |
| AUD-18 | OPEN | sandbox/report redaction | marker-only | GAP-RC-REDACTION | C015 | tenant uuid stripped | leak fixture | Negative leakage tests |
| AUD-19 | PARTIAL | manifest.py + JSON | 85 IDs wired | GAP-RC-SEMANTIC-GATES | C019 | closed-world 85 | extra ID | Exact 85 CheckIDs |
| AUD-20 | OPEN | digests.py | coarse FRZ-AUD-03 | GAP-RC-FRZ-MAP | C020,C021 | per-check FRZ | missing map | frz_coverage complete |
| AUD-21 | PARTIAL | modes.py compute_exit_code | L3 accepts {0,1,2,3} | GAP-RC-L3-FULL | C031,C074 | strict exit semantics | any exit pass | Deterministic exit codes |
| AUD-22 | CONFIRMED_OPEN | checks/executor.py | entrypoint exists not exec semantics | GAP-RC-SEMANTIC-GATES | C033-C036 | subprocess gate | is_file pass | 85 executed semantic gates |
| AUD-23 | CONFIRMED_OPEN | acceptance/test_ent_a_*.py | entrypoint_exists only | GAP-RC-SEMANTIC-GATES | C034-C036 | gate exit 0 | wiring only | Semantic oracle per check |
| AUD-24 | OPEN | integration/test_adversarial_adv01_30.py | scaffold not 31 faults | GAP-RC-ADV-INJECT | C037-C039 | per ADV inject | existence test | 31 fault injections |
| AUD-25 | FIXED_NOT_QUALIFIED | enterprise-audit.yml L38-41 | real FAST in CI | GAP-RC-PR-PATH-SKIP | C059 | CI artifact @ head | dry-run only | PR FAST proof URL |
| AUD-26 | EXTERNAL_BLOCKER | GitHub branch protection | OD-ENT-02 not verified | GAP-B BLK-CI-ENFORCE | C047 | N/A | N/A | Operator configures required checks |
| AUD-27 | OPEN | enterprise-audit-nightly.yml | FREEZE bootstrap risk | GAP-RC-NIGHTLY-FREEZE | C007,C046 | FULL only schedule | auto bootstrap | Operator-only FREEZE |
| AUD-28 | CONFIRMED_OPEN | audit.bat L10-14 | unknown arg → FAST | GAP-RC-LAUNCHER-BAT | C060,C063 | invalid exit 1 | silent FAST | Unknown mode fails |
| AUD-29 | PARTIAL | audit.bat L3-4 | repo root via script dir | GAP-RC-LAUNCHER-CWD | C063 | cwd≠repo | wrong cwd | Launch from any cwd |
| AUD-30 | OPEN | — | no spaced-path qual | GAP-RC-LAUNCHER-CWD | C063 | path w/ spaces | fail | Spaces in path proof |
| AUD-31 | OPEN | audit.bat L20 RANDOM | collision risk | GAP-RC-OUTPUT-COLLISION | C070 | parallel unique | clash | uuid+lock dirs |
| AUD-32 | OPEN | BAT vs SH | Windows-first dev | GAP-RC-LAUNCHER-SH | C061,C063 | Linux SH parity | divergence | Cross-platform parity |
| AUD-33 | PARTIAL | report.py | interrupt run_status open | GAP-RC-RUN-STATUS | C016 | partial INCOMPLETE | kill mid-run | run_status enum complete |
| AUD-34 | CLOSED_WITH_EVIDENCE | this master plan | HEAD 34e03425b reconciled | GAP-RC-SSOT-DRIFT | C075 | doc baseline updated | stale 1fa5347a5 header | Doc matches HEAD |
| AUD-35 | PARTIAL | Matrix D GAP register | 33 RC statuses @ HEAD | GAP-RC-SSOT-DRIFT | C055 | set equality | duplicate RC | Unique root causes |
| AUD-36 | EVIDENCE_GAP | ANNEX_G2 63 ENT rows | TEN/VBN in Annex C not G2 | GAP-RC-FRZ-MAP | C020,C055 | 85 row crosswalk | missing rows | G2+C matrix complete |
| AUD-37 | OPEN | ANNEX_H_WBS.md | stale vs C001-C075 | GAP-RC-SSOT-DRIFT | C055,C075 | H matches WBS | orphan task | Annex H reconciled |
| AUD-38 | CONFIRMED_OPEN | SHA ladder | no 34e03425b row yet | GAP-RC-SHA-LADDER | C048,C075 | append row | rewrite | Append-only ladder |
| AUD-39 | OPEN | PROTOCOL_03_DOSSIER.md | TES unset; 164 claim stale | GAP-RC-DOSSIER | C053,C075 | post-qual pins | fiction SHA | Dossier after qual only |
| AUD-40 | OPEN | FRZ-AUD-01..10 checklist | all OPEN | GAP-B multiple | C019-C064 | B6 closure | agent FRZ edit | Operator post-B6 |
| AUD-41 | EXTERNAL_BLOCKER | EVOL B6 pin PENDING | predecessor gate | GAP-B BLK-EVOL-B6 | — | N/A | N/A | External EVOL B6 |
| AUD-42 | OPEN | C057 doc | OD matrix not written | GAP-RC-SSOT-DRIFT | C057 | matrix published | N/A | Agent doc; operator signs separately |
| AUD-43 | OPEN | Roadmap §5 ENT row | FINAL not CLOSED | GAP-B BLK-ROADMAP-ENT-DEP | OD-ENT-00 | N/A | N/A | Operator roadmap §5 |
| AUD-44 | OPEN | ARCH-FREEZE handoff doc | blocked on ENT B6 | GAP-B BLK-ENT-B6 | C058 | handoff tree | early freeze | Post independent B6 |


---

## Matrix C — 75 TaskID WBS (dual-axis)

**Implementation:** NOT_STARTED · PARTIAL · IMPLEMENTED  
**Qualification:** NOT_QUALIFIED · QUALIFIED · EXTERNAL_PENDING  

**Counts @ `34e0342`:** implementation {'PARTIAL': 21, 'IMPLEMENTED': 4, 'NOT_STARTED': 50} · qualification {'NOT_QUALIFIED': 73, 'QUALIFIED': 2}

### Gate ownership (external — separate from TaskID axes)

| Gate | Owner | Blocks |
| --- | --- | --- |
| OD-ENT-00 / roadmap §5 | Operator | Program closure |
| OD-ENT-01 promotion | Operator | ACCEPTED_BASELINE_SHA |
| OD-ENT-02 required checks | Operator | BLK-CI-ENFORCE |
| OD-ENT-08 | Operator | Formal closure |
| Protocol 03 B6 | External auditor | B6_ACCEPTANCE_PUBLICATION_SHA |
| EVOL B6 predecessor | External | BLK-EVOL-B6 |

| TaskID | Batch | Original scope | Impl | Qual | Remaining | Deps | Primary test | Evidence | Acceptance | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | R0 | BaselinePhase FSM; candidate vs accepted paths | PARTIAL | NOT_QUALIFIED | Complete remaining scope: BaselinePhase FSM; candidate vs accepte | — | ENT-T-C001 in corrective plan | post-qual @ TEST_EXECUTION_SHA | BaselinePhase FSM; candidate vs accepted paths | ENT |
| C002 | R0 | Remove auto save_baseline; candidate write only on bootstrap | PARTIAL | NOT_QUALIFIED | Complete remaining scope: Remove auto save_baseline; candidate wr | C001 | ENT-T-C002 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Remove auto save_baseline; candidate write only on bootstrap | ENT |
| C003 | R0 | promote-baseline CLI; OD-ENT-01 verify boundary | PARTIAL | NOT_QUALIFIED | Complete remaining scope: promote-baseline CLI; OD-ENT-01 verify  | C001,C006 | ENT-T-C003 in corrective plan | post-qual @ TEST_EXECUTION_SHA | promote-baseline CLI; OD-ENT-01 verify boundary | ENT |
| C004 | R0 | certifying_run default true on manifest/report | IMPLEMENTED | QUALIFIED | None — qualified @ HEAD | — | ENT-T-C004 in corrective plan | unit/CLI @ 34e0342 | certifying_run default true on manifest/report | ENT |
| C005 | R0 | CLI --check-id requires --non-certifying | IMPLEMENTED | QUALIFIED | None — qualified @ HEAD | — | ENT-T-C005 in corrective plan | unit/CLI @ 34e0342 | CLI --check-id requires --non-certifying | ENT |
| C006 | R0 | Closed-world evidence artifact catalog + builder | PARTIAL | NOT_QUALIFIED | Complete remaining scope: Closed-world evidence artifact catalog  | — | ENT-T-C006 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Closed-world evidence artifact catalog + builder | ENT |
| C007 | R4 | Nightly workflow: no auto FREEZE bootstrap | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Nightly workflow: no auto FREEZE bootstrap | — | ENT-T-C007 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Nightly workflow: no auto FREEZE bootstrap | ENT |
| C008 | R0 | validate_domain_report() D01..D31 | NOT_STARTED | NOT_QUALIFIED | Implement full scope: validate_domain_report() D01..D31 | — | ENT-T-C008 in corrective plan | post-qual @ TEST_EXECUTION_SHA | validate_domain_report() D01..D31 | ENT |
| C009 | R1 | FREEZE FSM: bootstrap candidate vs normal FREEZE | PARTIAL | NOT_QUALIFIED | Complete remaining scope: FREEZE FSM: bootstrap candidate vs norm | C001,C002 | ENT-T-C009 in corrective plan | post-qual @ TEST_EXECUTION_SHA | FREEZE FSM: bootstrap candidate vs normal FREEZE | ENT |
| C010 | R1 | Fresh VBN domain report provenance | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Fresh VBN domain report provenance | — | ENT-T-C010 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Fresh VBN domain report provenance | ENT |
| C011 | R1 | domains_accounted<31 → BLOCKER on certifying FULL/FREEZE | NOT_STARTED | NOT_QUALIFIED | Implement full scope: domains_accounted<31 → BLOCKER on certifyin | C008 | ENT-T-C011 in corrective plan | post-qual @ TEST_EXECUTION_SHA | domains_accounted<31 → BLOCKER on certifying FULL/FREEZE | ENT |
| C012 | R1 | assert_git_ancestor for baseline/publish pins | PARTIAL | NOT_QUALIFIED | Complete remaining scope: assert_git_ancestor for baseline/publis | — | ENT-T-C012 in corrective plan | post-qual @ TEST_EXECUTION_SHA | assert_git_ancestor for baseline/publish pins | ENT |
| C013 | R1 | Atomic report publication temp+rename | PARTIAL | NOT_QUALIFIED | Complete remaining scope: Atomic report publication temp+rename | — | ENT-T-C013 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Atomic report publication temp+rename | ENT |
| C014 | R1 | Sandbox per-run isolation | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Sandbox per-run isolation | — | ENT-T-C014 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Sandbox per-run isolation | ENT |
| C015 | R1 | Log/report redaction v2 tenant-safe | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Log/report redaction v2 tenant-safe | — | ENT-T-C015 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Log/report redaction v2 tenant-safe | ENT |
| C016 | R1 | execution_manifest run_status on interrupt | NOT_STARTED | NOT_QUALIFIED | Implement full scope: execution_manifest run_status on interrupt | — | ENT-T-C016 in corrective plan | post-qual @ TEST_EXECUTION_SHA | execution_manifest run_status on interrupt | ENT |
| C017 | R1 | Sandbox timeout and orphan subprocess cleanup | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Sandbox timeout and orphan subprocess clean | — | ENT-T-C017 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Sandbox timeout and orphan subprocess cleanup | ENT |
| C018 | R1 | Runner enforces VBN rollup fail-closed | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Runner enforces VBN rollup fail-closed | C011 | ENT-T-C018 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Runner enforces VBN rollup fail-closed | ENT |
| C019 | R2 | Manifest closed-world vs Annex C 85 IDs | PARTIAL | NOT_QUALIFIED | Complete remaining scope: Manifest closed-world vs Annex C 85 IDs | — | ENT-T-C019 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Manifest closed-world vs Annex C 85 IDs | ENT |
| C020 | R2 | Per-check FRZ map JSON (not all FRZ-AUD-03) | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Per-check FRZ map JSON (not all FRZ-AUD-03) | — | ENT-T-C020 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Per-check FRZ map JSON (not all FRZ-AUD-03) | ENT |
| C021 | R2 | Manifest frz_mapping rows per check | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Manifest frz_mapping rows per check | — | ENT-T-C021 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Manifest frz_mapping rows per check | ENT |
| C022 | R2 | Per-check schema_version drift REGRESSED | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Per-check schema_version drift REGRESSED | — | ENT-T-C022 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Per-check schema_version drift REGRESSED | ENT |
| C023 | R2 | RESOLVED reopen when finding returns | NOT_STARTED | NOT_QUALIFIED | Implement full scope: RESOLVED reopen when finding returns | — | ENT-T-C023 in corrective plan | post-qual @ TEST_EXECUTION_SHA | RESOLVED reopen when finding returns | ENT |
| C024 | R2 | CHANGED-SINCE impact map data SSOT | NOT_STARTED | NOT_QUALIFIED | Implement full scope: CHANGED-SINCE impact map data SSOT | — | ENT-T-C024 in corrective plan | post-qual @ TEST_EXECUTION_SHA | CHANGED-SINCE impact map data SSOT | ENT |
| C025 | R2 | Unknown changed paths fail-closed BLOCKER | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Unknown changed paths fail-closed BLOCKER | — | ENT-T-C025 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Unknown changed paths fail-closed BLOCKER | ENT |
| C026 | R2 | verify_closed_evidence_bundle tamper-closed | PARTIAL | NOT_QUALIFIED | Complete remaining scope: verify_closed_evidence_bundle tamper-cl | C006 | ENT-T-C026 in corrective plan | post-qual @ TEST_EXECUTION_SHA | verify_closed_evidence_bundle tamper-closed | ENT |
| C027 | R2 | Evidence path traversal rejection | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Evidence path traversal rejection | — | ENT-T-C027 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Evidence path traversal rejection | ENT |
| C028 | R2 | manifest_digest change invalidates accepted baseline | NOT_STARTED | NOT_QUALIFIED | Implement full scope: manifest_digest change invalidates accepted | — | ENT-T-C028 in corrective plan | post-qual @ TEST_EXECUTION_SHA | manifest_digest change invalidates accepted baseline | ENT |
| C029 | R2 | Impact map generator build_impact_map.py | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Impact map generator build_impact_map.py | — | ENT-T-C029 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Impact map generator build_impact_map.py | ENT |
| C030 | R2 | Manifest DAG orphan/cycle detection | PARTIAL | NOT_QUALIFIED | Complete remaining scope: Manifest DAG orphan/cycle detection | — | ENT-T-C030 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Manifest DAG orphan/cycle detection | ENT |
| C031 | R3 | L3 real certifying FULL 85/85 CLI subprocess | NOT_STARTED | NOT_QUALIFIED | Implement full scope: L3 real certifying FULL 85/85 CLI subproces | C004,C011 | ENT-T-C031 in corrective plan | post-qual @ TEST_EXECUTION_SHA | L3 real certifying FULL 85/85 CLI subprocess | ENT |
| C032 | R3 | L3 FREEZE pair: first candidate + normal FREEZE | NOT_STARTED | NOT_QUALIFIED | Implement full scope: L3 FREEZE pair: first candidate + normal FR | C002,C009,C011,C031 | ENT-T-C032 in corrective plan | post-qual @ TEST_EXECUTION_SHA | L3 FREEZE pair: first candidate + normal FREEZE | ENT |
| C033 | R3 | acceptance gate helper conftest | NOT_STARTED | NOT_QUALIFIED | Implement full scope: acceptance gate helper conftest | C004 | ENT-T-C033 in corrective plan | post-qual @ TEST_EXECUTION_SHA | acceptance gate helper conftest | ENT |
| C034 | R3 | ENT-A 63 semantic subprocess gates | NOT_STARTED | NOT_QUALIFIED | Implement full scope: ENT-A 63 semantic subprocess gates | C033 | ENT-T-C034 in corrective plan | post-qual @ TEST_EXECUTION_SHA | ENT-A 63 semantic subprocess gates | ENT |
| C035 | R3 | TEN 13 semantic subprocess gates | NOT_STARTED | NOT_QUALIFIED | Implement full scope: TEN 13 semantic subprocess gates | C033 | ENT-T-C035 in corrective plan | post-qual @ TEST_EXECUTION_SHA | TEN 13 semantic subprocess gates | ENT |
| C036 | R3 | VBN 9 semantic subprocess gates | NOT_STARTED | NOT_QUALIFIED | Implement full scope: VBN 9 semantic subprocess gates | C033 | ENT-T-C036 in corrective plan | post-qual @ TEST_EXECUTION_SHA | VBN 9 semantic subprocess gates | ENT |
| C037 | R3 | ADV-01..10 fault injection suite | NOT_STARTED | NOT_QUALIFIED | Implement full scope: ADV-01..10 fault injection suite | C004,C026 | ENT-T-C037 in corrective plan | post-qual @ TEST_EXECUTION_SHA | ADV-01..10 fault injection suite | ENT |
| C038 | R3 | ADV-11..20 fault injection suite | NOT_STARTED | NOT_QUALIFIED | Implement full scope: ADV-11..20 fault injection suite | C037 | ENT-T-C038 in corrective plan | post-qual @ TEST_EXECUTION_SHA | ADV-11..20 fault injection suite | ENT |
| C039 | R3 | ADV-21..31 fault injection suite | NOT_STARTED | NOT_QUALIFIED | Implement full scope: ADV-21..31 fault injection suite | C037 | ENT-T-C039 in corrective plan | post-qual @ TEST_EXECUTION_SHA | ADV-21..31 fault injection suite | ENT |
| C040 | R3 | pytest markers ent_l1/l2/l3_heavy taxonomy | NOT_STARTED | NOT_QUALIFIED | Implement full scope: pytest markers ent_l1/l2/l3_heavy taxonomy | — | ENT-T-C040 in corrective plan | post-qual @ TEST_EXECUTION_SHA | pytest markers ent_l1/l2/l3_heavy taxonomy | ENT |
| C041 | R3 | Mechanical pytest collect-only inventory reconcile | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Mechanical pytest collect-only inventory re | — | ENT-T-C041 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Mechanical pytest collect-only inventory reconcile | ENT |
| C042 | R3 | Promotion trust tests + bundle leg after C026 | PARTIAL | NOT_QUALIFIED | Complete remaining scope: Promotion trust tests + bundle leg afte | C003,C026 | ENT-T-C042 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Promotion trust tests + bundle leg after C026 | ENT |
| C043 | R3 | Forged vbn_domains.v1.json detection | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Forged vbn_domains.v1.json detection | — | ENT-T-C043 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Forged vbn_domains.v1.json detection | ENT |
| C044 | R3 | Report status tamper audit-of-auditor | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Report status tamper audit-of-auditor | — | ENT-T-C044 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Report status tamper audit-of-auditor | ENT |
| C045 | R4 | enterprise-audit.yml skeleton identity | PARTIAL | NOT_QUALIFIED | Complete remaining scope: enterprise-audit.yml skeleton identity | — | ENT-T-C045 in corrective plan | post-qual @ TEST_EXECUTION_SHA | enterprise-audit.yml skeleton identity | ENT |
| C046 | R4 | Nightly align with C007 operator-only FREEZE | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Nightly align with C007 operator-only FREEZ | — | ENT-T-C046 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Nightly align with C007 operator-only FREEZE | ENT |
| C047 | R4 | CI runbook required-check procedure (no false enforce claim) | NOT_STARTED | NOT_QUALIFIED | Implement full scope: CI runbook required-check procedure (no fal | — | ENT-T-C047 in corrective plan | post-qual @ TEST_EXECUTION_SHA | CI runbook required-check procedure (no false enforce claim) | ENT |
| C048 | R5 | SHA ladder append-only rules + IMPLEMENTATION rows | NOT_STARTED | NOT_QUALIFIED | Implement full scope: SHA ladder append-only rules + IMPLEMENTATI | — | ENT-T-C048 in corrective plan | post-qual @ TEST_EXECUTION_SHA | SHA ladder append-only rules + IMPLEMENTATION rows | ENT |
| C049 | R5 | Evidence env fingerprint artifact | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Evidence env fingerprint artifact | — | ENT-T-C049 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Evidence env fingerprint artifact | ENT |
| C050 | R5 | build_evidence_bundle.py orchestrator | NOT_STARTED | NOT_QUALIFIED | Implement full scope: build_evidence_bundle.py orchestrator | C026,C031 | ENT-T-C050 in corrective plan | post-qual @ TEST_EXECUTION_SHA | build_evidence_bundle.py orchestrator | ENT |
| C051 | R5 | run_qual_wave.py L1→L3 orchestrator | NOT_STARTED | NOT_QUALIFIED | Implement full scope: run_qual_wave.py L1→L3 orchestrator | C031,C032 | ENT-T-C051 in corrective plan | post-qual @ TEST_EXECUTION_SHA | run_qual_wave.py L1→L3 orchestrator | ENT |
| C052 | R5 | Runbook retention/replay CI evidence | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Runbook retention/replay CI evidence | — | ENT-T-C052 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Runbook retention/replay CI evidence | ENT |
| C053 | R5 | Protocol 03 dossier mechanical fill post-qual | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Protocol 03 dossier mechanical fill post-qu | — | ENT-T-C053 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Protocol 03 dossier mechanical fill post-qual | ENT |
| C054 | R5 | mechanical_readiness_check.py fail-closed pre-B6 | PARTIAL | NOT_QUALIFIED | Complete remaining scope: mechanical_readiness_check.py fail-clos | — | ENT-T-C054 in corrective plan | post-qual @ TEST_EXECUTION_SHA | mechanical_readiness_check.py fail-closed pre-B6 | ENT |
| C055 | R5 | Annex F gap register reconciliation @ HEAD | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Annex F gap register reconciliation @ HEAD | — | ENT-T-C055 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Annex F gap register reconciliation @ HEAD | ENT |
| C056 | R5 | Corrective plan supersede header note | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Corrective plan supersede header note | — | ENT-T-C056 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Corrective plan supersede header note | ENT |
| C057 | R5 | Operator OD-ENT decision authority matrix (doc only; agent writes) | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Operator OD-ENT decision authority matrix ( | — | ENT-T-C057 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Operator OD-ENT decision authority matrix (doc only; agent w | ENT |
| C058 | R5 | B6 handoff replay tree layout validator | NOT_STARTED | NOT_QUALIFIED | Implement full scope: B6 handoff replay tree layout validator | C050,C053 | ENT-T-C058 in corrective plan | post-qual @ TEST_EXECUTION_SHA | B6 handoff replay tree layout validator | ENT |
| C059 | R4 | PR FAST real certifying run @ PR head SHA | PARTIAL | NOT_QUALIFIED | Complete remaining scope: PR FAST real certifying run @ PR head S | C004,C045 | ENT-T-C059 in corrective plan | post-qual @ TEST_EXECUTION_SHA | PR FAST real certifying run @ PR head SHA | ENT |
| C060 | R4 | audit.bat modes + fail unknown args + unique out dir | PARTIAL | NOT_QUALIFIED | Complete remaining scope: audit.bat modes + fail unknown args + u | C004,C062 | ENT-T-C060 in corrective plan | post-qual @ TEST_EXECUTION_SHA | audit.bat modes + fail unknown args + unique out dir | ENT |
| C061 | R4 | audit.sh parity with BAT | PARTIAL | NOT_QUALIFIED | Complete remaining scope: audit.sh parity with BAT | C060,C062 | ENT-T-C061 in corrective plan | post-qual @ TEST_EXECUTION_SHA | audit.sh parity with BAT | ENT |
| C062 | R4 | CLI/report summary UX fields | PARTIAL | NOT_QUALIFIED | Complete remaining scope: CLI/report summary UX fields | — | ENT-T-C062 in corrective plan | post-qual @ TEST_EXECUTION_SHA | CLI/report summary UX fields | ENT |
| C063 | R4 | Launcher integration tests matrix (16 cases) | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Launcher integration tests matrix (16 cases | C059,C060,C061 | ENT-T-C063 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Launcher integration tests matrix (16 cases) | ENT |
| C064 | R5 | Operator quickstart + FRZ-AUD-10 mapping doc | NOT_STARTED | NOT_QUALIFIED | Implement full scope: Operator quickstart + FRZ-AUD-10 mapping do | — | ENT-T-C064 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Operator quickstart + FRZ-AUD-10 mapping doc | ENT |
| C065 | R0 | ECAP OD-ENT-01 verify: ladder pins, operator_records path, dossier | PARTIAL | NOT_QUALIFIED | Mandatory verify_closed_bundle on promote; e2e promote w/ ladder  | C003,C026 | ENT-T-C065 in corrective plan | post-qual @ TEST_EXECUTION_SHA | ECAP OD-ENT-01 verify: ladder pins, operator_records path, d | ENT |
| C066 | R5 | Readiness parses qual artifacts @ TEST_EXECUTION_SHA (not file exists) | PARTIAL | NOT_QUALIFIED | Wire TEN/VBN/ADV/CI/bundle parsers; remove hardcoded False stubs  | C031,C032,C051,C074 | ENT-T-C066 in corrective plan | post-qual @ TEST_EXECUTION_SHA | Readiness parses qual artifacts @ TEST_EXECUTION_SHA (not fi | ENT |
| C067 | R0 | dry_run non-certifying; INCOMPLETE exit; block qual bundle | IMPLEMENTED | NOT_QUALIFIED | Complete remaining scope: dry_run non-certifying; INCOMPLETE exit | — | ENT-T-C067 in corrective plan | post-qual @ TEST_EXECUTION_SHA | dry_run non-certifying; INCOMPLETE exit; block qual bundle | ENT |
| C068 | R0 | build_evidence_manifest requires execution manifest eligibility | PARTIAL | NOT_QUALIFIED | Require execution_manifest on all certifying bundle builds; rejec | C067,C006 | ENT-T-C068 in corrective plan | post-qual @ TEST_EXECUTION_SHA | build_evidence_manifest requires execution manifest eligibil | ENT |
| C069 | R0 | jsonschema draft 2020-12 nested validators all artifacts | NOT_STARTED | NOT_QUALIFIED | Implement full scope: jsonschema draft 2020-12 nested validators  | — | ENT-T-C069 in corrective plan | post-qual @ TEST_EXECUTION_SHA | jsonschema draft 2020-12 nested validators all artifacts | ENT |
| C070 | R4 | uuid+timestamp+lock collision-safe output dirs | NOT_STARTED | NOT_QUALIFIED | Implement full scope: uuid+timestamp+lock collision-safe output d | — | ENT-T-C070 in corrective plan | post-qual @ TEST_EXECUTION_SHA | uuid+timestamp+lock collision-safe output dirs | ENT |
| C071 | R1 | aggregate_status excludes RESOLVED from blocking severities | IMPLEMENTED | NOT_QUALIFIED | Complete remaining scope: aggregate_status excludes RESOLVED from | — | ENT-T-C071 in corrective plan | post-qual @ TEST_EXECUTION_SHA | aggregate_status excludes RESOLVED from blocking severities | ENT |
| C072 | R2 | CHANGED-SINCE semantic impact + ancestry pins | NOT_STARTED | NOT_QUALIFIED | Implement full scope: CHANGED-SINCE semantic impact + ancestry pi | — | ENT-T-C072 in corrective plan | post-qual @ TEST_EXECUTION_SHA | CHANGED-SINCE semantic impact + ancestry pins | ENT |
| C073 | R0 | evidence_artifact_catalog mode profiles FAST/FULL/FREEZE/B6 | NOT_STARTED | NOT_QUALIFIED | Implement full scope: evidence_artifact_catalog mode profiles FAS | — | ENT-T-C073 in corrective plan | post-qual @ TEST_EXECUTION_SHA | evidence_artifact_catalog mode profiles FAST/FULL/FREEZE/B6 | ENT |
| C074 | R3 | test_l3_qual_wave.py clean-checkout strict harness | NOT_STARTED | NOT_QUALIFIED | Implement full scope: test_l3_qual_wave.py clean-checkout strict  | C031,C032 | ENT-T-C074 in corrective plan | post-qual @ TEST_EXECUTION_SHA | test_l3_qual_wave.py clean-checkout strict harness | ENT |
| C075 | R5 | SSOT reconcile this doc + annexes + ladder + dossier counts | PARTIAL | NOT_QUALIFIED | Annex F/H/G2 header sync; append ladder IMPLEMENTATION_SHA row (n | C041,C048,C055 | ENT-T-C075 in corrective plan | post-qual @ TEST_EXECUTION_SHA | SSOT reconcile this doc + annexes + ladder + dossier counts | ENT |

---

## Matrix D — GAP reconciliation

### D.1 GAP-RC (33) @ HEAD

| GAP-ID | Source | Root cause | Sev | Code | Status @ HEAD | TaskID |
| --- | --- | --- | --- | --- | --- | --- |
| GAP-RC-OD-TRUST | A-01,P0-01,F-07 | OD verify incomplete vs ECAP | CRIT | baseline.py | PARTIAL | C003,C042,C065 |
| GAP-RC-MECH-READY | A-02,A-20 | Readiness not tied to qual artifacts | CRIT | mechanical_readiness_check.py | PARTIAL | C054,C066 |
| GAP-RC-EVIDENCE-TRUST | A-03,A-06,A-14 | Bundle/manifest coupling partial | CRIT | evidence.py | PARTIAL | C006,C026,C068,C073 |
| GAP-RC-L3-FULL | A-04,P0-02 | No real 85/85 L3 | CRIT | l3/ | OPEN | C031,C032,C074 |
| GAP-RC-DRY-RUN | A-05,ADV-05 | dry_run policy landed; not qual | CRIT | runner.py | PARTIAL | C067 |
| GAP-RC-VBN-31 | A-07,A-08 | D01..D31 validator open | CRIT | vbn.py | OPEN | C008-C011,C018,C043 |
| GAP-RC-CHANGED-IMPACT | A-09,P1-03 | Path-only CHANGED-SINCE | MAJ | changed_since.py | OPEN | C024,C025,C029,C072 |
| GAP-RC-GIT-ANCESTRY | A-10,P1-13 | Ancestry partial | MAJ | repo_identity.py | PARTIAL | C012,C072 |
| GAP-RC-RESOLVED-BLOCK | A-11 | aggregate filter landed; test gap | MAJ | modes.py | PARTIAL | C071,C023 |
| GAP-RC-DRIFT-SCHEMA | A-12,P1-02 | schema_version drift open | MAJ | baseline.py | OPEN | C022,C028 |
| GAP-RC-JSON-SCHEMA | A-13 | Shallow validator | MAJ | validators.py | OPEN | C069 |
| GAP-RC-ATOMIC-PUB | A-15 | Per-file atomic only | MAJ | report.py | PARTIAL | C013 |
| GAP-RC-SANDBOX | A-16,P1-10 | Isolation/timeouts | MAJ | sandbox.py | OPEN | C014-C017 |
| GAP-RC-REDACTION | A-17,P1-11 | Marker redaction only | MAJ | sandbox.py | OPEN | C015 |
| GAP-RC-SEMANTIC-GATES | A-18 | Gates not semantic | CRIT | checks/executor.py | OPEN | C033-C036 |
| GAP-RC-ADV-INJECT | A-19,P1-04 | ADV scaffold only | MAJ | integration/adversarial | OPEN | C037-C039 |
| GAP-RC-PR-PATH-SKIP | A-22,P1-05 | paths-filter; real FAST landed | MAJ | enterprise-audit.yml | PARTIAL | C046,C059 |
| GAP-RC-NIGHTLY-FREEZE | A-23,P1-06 | Nightly FREEZE bootstrap | MAJ | nightly yml | OPEN | C007,C046 |
| GAP-RC-LAUNCHER-BAT | A-24 | Unknown arg → FAST | MAJ | audit.bat | OPEN | C060,C063 |
| GAP-RC-LAUNCHER-SH | A-25 | SH parity gaps | MAJ | audit.sh | OPEN | C061,C063 |
| GAP-RC-LAUNCHER-CWD | A-26 | Launcher cwd/spaces qual | MAJ | launchers | OPEN | C063 |
| GAP-RC-OUTPUT-COLLISION | A-27 | RANDOM output dirs | MAJ | audit.bat | OPEN | C070 |
| GAP-RC-DOSSIER | A-28 | Draft dossier unpinned | MAJ | dossier md | OPEN | C053,C075 |
| GAP-RC-SHA-LADDER | A-29,P1-17 | Ladder stale vs HEAD | MAJ | SHA ladder | OPEN | C048,C075 |
| GAP-RC-SSOT-DRIFT | A-30,P1-16 | Doc counts drift | MAJ | annexes | PARTIAL | C041,C055,C064,C075 |
| GAP-RC-FRZ-MAP | P1-08 | FRZ artifact coarse | MAJ | digests.py | OPEN | C020,C021 |
| GAP-RC-RUN-STATUS | P1-14 | run_status on interrupt | MAJ | runner.py | OPEN | C016 |
| GAP-RC-REQUAL-DIGEST | requal | manifest_digest change | MAJ | baseline.py | OPEN | C028 |
| GAP-RC-AUDIT-OF-AUDITOR | false PASS | status tamper | MAJ | report | OPEN | C044 |
| GAP-RC-EVID-PUBLISH | CP-08 | Orchestrated publish | MAJ | scripts | OPEN | C049-C051,C058 |
| GAP-RC-CI-PROFILE | GAP-P-01 | profile identity | MIN | yaml | OPEN | C045 |
| GAP-RC-CERTIFY-SPLIT | P0-04 | closed | — | contracts | CLOSED_WITH_EVIDENCE | C004,C005 |
| GAP-RC-CANDIDATE-FSM | P0-01 | candidate FSM partial | — | baseline | PARTIAL | C001,C002 |

### D.2 GAP-B (6)

| GAP-B-ID | Owner |
| --- | --- |
| BLK-EVOL-B6 | Auditor + operator |
| BLK-EVOL-FRZ | Operator |
| BLK-EVOL-HANDOFF | Operator |
| BLK-CI-ENFORCE | OD-ENT-02 + GitHub |
| BLK-ROADMAP-ENT-DEP | OD-ENT-00 |
| BLK-ENT-B6 | Protocol 03 |

### D.3 GAP-COV (85)

One `GAP-COV-<CheckID>` per manifest row — all **OPEN** until Matrix E semantic closure @ `TEST_EXECUTION_SHA`.

**Set equality:** |GAP-COV| = |CheckID| = 85.

### D.4 Annex F historical buckets (136 rows — not merged into 32 RC)

Reconcile in **C055** without duplicating GAP-RC root causes.

---

## Matrix E — 85 CheckID

| CheckID | Invariant family | Entrypoint | Current oracle | Missing work | Negative test | FRZ | Task | Evidence | Qual SHA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ENT-A-BND-01 | Annex C | `pytest_nodeid` tests/unit/runtime/architecture/test_hardening_3_layer_ | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-BND-01 OPEN | *unset* |
| ENT-A-BND-02 | Annex C | `pytest_nodeid` tests/unit/runtime/architecture/test_hardening_3_layer_ | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-BND-02 OPEN | *unset* |
| ENT-A-BND-03 | Annex C | `python_script` scripts/maintenance/check_agents_no_tier3_imports.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-BND-03 OPEN | *unset* |
| ENT-A-OWN-01 | Annex C | `pytest_file` tests/unit/architecture/test_ebh_3_dependency_ownership | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-OWN-01 OPEN | *unset* |
| ENT-A-OWN-02 | Annex C | `pytest_file` tests/unit/architecture/test_ebh_3_dependency_ownership | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-OWN-02 OPEN | *unset* |
| ENT-A-OWN-03 | Annex C | `pytest_file` tests/unit/architecture/test_ebh_3_dependency_ownership | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-OWN-03 OPEN | *unset* |
| ENT-A-CTR-01 | Annex C | `pytest_nodeid` tests/unit/architecture/test_ebh_2i_final_rescan_gate.p | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-CTR-01 OPEN | *unset* |
| ENT-A-CTR-02 | Annex C | `pytest_nodeid` tests/unit/architecture/test_ebh_2i_final_rescan_gate.p | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-CTR-02 OPEN | *unset* |
| ENT-A-CTR-03 | Annex C | `pytest_nodeid` tests/unit/architecture/test_ebh_2i_final_rescan_gate.p | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-CTR-03 OPEN | *unset* |
| ENT-A-TYP-01 | Annex C | `pytest_file` tests/unit/architecture/test_ebh_2a_public_contract_bou | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-TYP-01 OPEN | *unset* |
| ENT-A-TYP-02 | Annex C | `pytest_nodeid` tests/unit/architecture/test_ebh_2i_final_rescan_gate.p | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-TYP-02 OPEN | *unset* |
| ENT-A-TYP-03 | Annex C | `pytest_nodeid` tests/unit/architecture/test_ebh_2i_final_rescan_gate.p | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-TYP-03 OPEN | *unset* |
| ENT-A-DUP-01 | Annex C | `pytest_file` tests/qualification/dup_x/test_dup_x_mechanical_exit_ga | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-DUP-01 OPEN | *unset* |
| ENT-A-DUP-02 | Annex C | `pytest_file` tests/qualification/dup_x/test_dup_x_mechanical_exit_ga | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-DUP-02 OPEN | *unset* |
| ENT-A-DUP-03 | Annex C | `pytest_file` tests/qualification/dup_x/test_dup_x_mechanical_exit_ga | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-DUP-03 OPEN | *unset* |
| ENT-A-PLG-01 | Annex C | `pytest_file` tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pa | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-PLG-01 OPEN | *unset* |
| ENT-A-PLG-02 | Annex C | `pytest_file` tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pa | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-PLG-02 OPEN | *unset* |
| ENT-A-PLG-03 | Annex C | `pytest_file` tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pa | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-PLG-03 OPEN | *unset* |
| ENT-A-VBN-01 | Annex C | `pytest_nodeid` tests/qualification/ebh_5/test_ebh_5_g06_enterprise_int | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-VBN-01 OPEN | *unset* |
| ENT-A-VBN-02 | Annex C | `pytest_nodeid` tests/qualification/ebh_6/test_ebh_6_hardcoding_registe | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-VBN-02 OPEN | *unset* |
| ENT-A-VBN-03 | Annex C | `pytest_file` tests/qualification/config_x/test_config_x_final_reconc | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-VBN-03 OPEN | *unset* |
| ENT-A-GOV-01 | Annex C | `pytest_nodeid` tests/qualification/governance/gov_x2/test_gov_x2_quali | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-GOV-01 OPEN | *unset* |
| ENT-A-GOV-02 | Annex C | `pytest_nodeid` tests/qualification/governance/gov_x2/test_gov_x2_quali | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-GOV-02 OPEN | *unset* |
| ENT-A-GOV-03 | Annex C | `pytest_nodeid` tests/qualification/governance/gov_x2/test_gov_x2_proof | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-GOV-03 OPEN | *unset* |
| ENT-A-EXE-01 | Annex C | `pytest_file` tests/unit/runtime/architecture/test_ee_a1_execution_en | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-EXE-01 OPEN | *unset* |
| ENT-A-EXE-02 | Annex C | `pytest_file` tests/unit/runtime/architecture/test_ee_a1_execution_en | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-EXE-02 OPEN | *unset* |
| ENT-A-EXE-03 | Annex C | `pytest_file` tests/unit/runtime/architecture/test_ee_a1_execution_en | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-EXE-03 OPEN | *unset* |
| ENT-A-OBS-01 | Annex C | `pytest_nodeid` tests/unit/runtime/architecture/test_harness_w5_typed_o | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-OBS-01 OPEN | *unset* |
| ENT-A-OBS-02 | Annex C | `pytest_nodeid` tests/unit/runtime/architecture/test_harness_w5_typed_o | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-OBS-02 OPEN | *unset* |
| ENT-A-OBS-03 | Annex C | `pytest_nodeid` tests/qualification/trace_x/test_trace_x_p4_r1_recorder | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-OBS-03 OPEN | *unset* |
| ENT-A-STA-01 | Annex C | `pytest_nodeid` tests/qualification/state_x/test_state_x_p0_baseline.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-STA-01 OPEN | *unset* |
| ENT-A-STA-02 | Annex C | `pytest_nodeid` tests/qualification/state_x/test_state_x_p0_baseline.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-STA-02 OPEN | *unset* |
| ENT-A-STA-03 | Annex C | `pytest_nodeid` tests/qualification/state_x/test_state_x_r3_auxiliary_d | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-STA-03 OPEN | *unset* |
| ENT-A-SEC-01 | Annex C | `python_script` scripts/maintenance/check_immutable_security_audit_trai | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-SEC-01 OPEN | *unset* |
| ENT-A-SEC-02 | Annex C | `python_script` scripts/maintenance/check_immutable_security_audit_trai | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-SEC-02 OPEN | *unset* |
| ENT-A-SEC-03 | Annex C | `pytest_nodeid` tests/qualification/ent_audit_x/unit/test_enterprise_au | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-SEC-03 OPEN | *unset* |
| ENT-A-REL-01 | Annex C | `pytest_nodeid` tests/unit/runtime/architecture/test_harness_w4_r1_prod | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REL-01 OPEN | *unset* |
| ENT-A-REL-02 | Annex C | `pytest_nodeid` tests/qualification/harness_01/test_harness_01_gates.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REL-02 OPEN | *unset* |
| ENT-A-REL-03 | Annex C | `pytest_nodeid` tests/qualification/harness_01/test_harness_01_gates.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REL-03 OPEN | *unset* |
| ENT-A-REC-01 | Annex C | `pytest_nodeid` tests/qualification/state_x/test_state_x_p0_baseline.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REC-01 OPEN | *unset* |
| ENT-A-REC-02 | Annex C | `pytest_nodeid` tests/qualification/state_x/test_state_x_r3_auxiliary_d | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REC-02 OPEN | *unset* |
| ENT-A-REC-03 | Annex C | `pytest_nodeid` tests/qualification/state_x/test_state_x_r3_auxiliary_d | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REC-03 OPEN | *unset* |
| ENT-A-PRD-01 | Annex C | `python_script` scripts/gates/check_application_production_gates.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-PRD-01 OPEN | *unset* |
| ENT-A-PRD-02 | Annex C | `python_script` scripts/gates/check_application_production_gates.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-PRD-02 OPEN | *unset* |
| ENT-A-PRD-03 | Annex C | `pytest_nodeid` tests/qualification/ent_audit_x/unit/test_report_includ | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-PRD-03 OPEN | *unset* |
| ENT-A-CMP-01 | Annex C | `pytest_nodeid` tests/qualification/compat_x/test_compat_x_inventory_ga | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-CMP-01 OPEN | *unset* |
| ENT-A-CMP-02 | Annex C | `pytest_nodeid` tests/qualification/compat_x/test_compat_x_persisted_r2 | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-CMP-02 OPEN | *unset* |
| ENT-A-CMP-03 | Annex C | `pytest_nodeid` tests/qualification/compat_x/test_compat_x_inventory_ga | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-CMP-03 OPEN | *unset* |
| ENT-A-REG-01 | Annex C | `pytest_nodeid` tests/qualification/roadmap_replay_x/test_roadmap_repla | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REG-01 OPEN | *unset* |
| ENT-A-REG-02 | Annex C | `pytest_file` tests/qualification/evol_x/gates/test_evol_x_closed_wor | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REG-02 OPEN | *unset* |
| ENT-A-REG-03 | Annex C | `python_script` scripts/gates/check_application_production_gates.py | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-REG-03 OPEN | *unset* |
| ENT-A-DOC-01 | Annex C | `pytest_nodeid` tests/qualification/roadmap_replay_x/test_roadmap_repla | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-DOC-01 OPEN | *unset* |
| ENT-A-DOC-02 | Annex C | `python_script` scripts/enterprise_audit/check_manifest_annex_c_alignme | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-DOC-02 OPEN | *unset* |
| ENT-A-DOC-03 | Annex C | `pytest_nodeid` tests/qualification/roadmap_replay_x/test_roadmap_repla | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-DOC-03 OPEN | *unset* |
| ENT-A-TEN-01 | Annex C | `pytest_file` tests/qualification/tenant_x/test_tenant_x_final_curren | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-TEN-01 OPEN | *unset* |
| ENT-A-TEN-02 | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r1_current_h | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-TEN-02 OPEN | *unset* |
| ENT-A-TEN-03 | Annex C | `pytest_nodeid` tests/qualification/ent_audit_x/integration/test_ent_a_ | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-TEN-03 OPEN | *unset* |
| ENT-A-EVOL-01 | Annex C | `pytest_file` tests/qualification/evol_x/gates/test_evol_x_selection. | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-EVOL-01 OPEN | *unset* |
| ENT-A-EVOL-02 | Annex C | `pytest_file` tests/qualification/evol_x/gates/test_evol_x_selection_ | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-EVOL-02 OPEN | *unset* |
| ENT-A-EVOL-03 | Annex C | `pytest_file` tests/qualification/evol_x/gates/test_evol_x_closed_wor | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-EVOL-03 OPEN | *unset* |
| ENT-A-INV-01 | Annex C | `pytest_file` tests/qualification/evol_x/gates/test_evol_x_closed_wor | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-INV-01 OPEN | *unset* |
| ENT-A-INV-02 | Annex C | `pytest_nodeid` tests/qualification/ent_audit_x/unit/test_frz_coverage_ | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-INV-02 OPEN | *unset* |
| ENT-A-INV-03 | Annex C | `pytest_nodeid` tests/qualification/ent_audit_x/integration/test_ent_au | entrypoint_exists / executor skip | semantic gate @ TES | architecture inject | Annex C/G2 | C034 | GAP-COV-ENT-A-INV-03 OPEN | *unset* |
| TEN-IDENTITY | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r1_current_h | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-IDENTITY OPEN | *unset* |
| TEN-PROPAGATION | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_p0_current_h | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-PROPAGATION OPEN | *unset* |
| TEN-AUTHORITY | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r4_fast.py:: | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-AUTHORITY OPEN | *unset* |
| TEN-STATE | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r4_fast.py:: | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-STATE OPEN | *unset* |
| TEN-PROVIDER | Annex C | `pytest_nodeid` tests/qualification/existing_capability_configuration/t | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-PROVIDER OPEN | *unset* |
| TEN-CONFIG | Annex C | `pytest_file` tests/qualification/config_x/test_config_x_final_reconc | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-CONFIG OPEN | *unset* |
| TEN-CREDENTIALS | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r2_current_h | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-CREDENTIALS OPEN | *unset* |
| TEN-EVIDENCE | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r4_fast.py:: | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-EVIDENCE OPEN | *unset* |
| TEN-TRACE | Annex C | `pytest_nodeid` tests/qualification/trace_x/test_trace_x_p4_r1_recorder | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-TRACE OPEN | *unset* |
| TEN-RECOVERY | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r4_fast.py:: | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-RECOVERY OPEN | *unset* |
| TEN-ASYNC | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r4_fast.py:: | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-ASYNC OPEN | *unset* |
| TEN-CROSS-TENANT | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r4_fast.py:: | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-CROSS-TENANT OPEN | *unset* |
| TEN-FAIL-CLOSED | Annex C | `pytest_nodeid` tests/qualification/tenant_x/test_tenant_x_r4_fast.py:: | entrypoint_exists / executor skip | semantic gate @ TES | tenant leak inject | FRZ-TEN-* | C035 | GAP-COV-TEN-FAIL-CLOSED OPEN | *unset* |
| VBN-INVENTORY-DRIFT | Annex C | `pytest_nodeid` tests/qualification/ebh_5/test_ebh_5_g06_enterprise_int | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-INVENTORY-DRIFT OPEN | *unset* |
| VBN-CORE-HARDCODING | Annex C | `pytest_nodeid` tests/qualification/ebh_6/test_ebh_6_hardcoding_registe | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-CORE-HARDCODING OPEN | *unset* |
| VBN-SHADOW-SELECTION | Annex C | `pytest_file` tests/qualification/config_x/test_config_x_final_reconc | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-SHADOW-SELECTION OPEN | *unset* |
| VBN-SILENT-FALLBACK | Annex C | `pytest_file` tests/qualification/ebh_6/test_ebh_6_config_x_head_reco | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-SILENT-FALLBACK OPEN | *unset* |
| VBN-CONTRACT-LEAKAGE | Annex C | `pytest_nodeid` tests/unit/architecture/test_ebh_2i_final_rescan_gate.p | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-CONTRACT-LEAKAGE OPEN | *unset* |
| VBN-SQLITE-EFFECTIVE | Annex C | `pytest_file` tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-SQLITE-EFFECTIVE OPEN | *unset* |
| VBN-OBS-OPS-COUPLING | Annex C | `pytest_nodeid` tests/unit/runtime/architecture/test_harness_w5_typed_o | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-OBS-OPS-COUPLING OPEN | *unset* |
| VBN-REPLACEABILITY-MATRIX | Annex C | `pytest_file` tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_op | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-REPLACEABILITY-MATRIX OPEN | *unset* |
| VBN-REACHABILITY | Annex C | `pytest_file` tests/qualification/ebh_5/test_ebh_5_behavioral_package | entrypoint_exists / executor skip | semantic gate @ TES | VBN violation inject | FRZ-RPL-* | C036 | GAP-COV-VBN-REACHABILITY OPEN | *unset* |

**TEN13:** CheckIDs `TEN-*` (13) · **VBN9:** `VBN-*` (9) · **D01..D31:** Annex D rollup via C008–C011.

---

## Matrix F — ADV-01..31

| ADV-ID | Threat | Fault injection | Expected finding/exit | Test path | TaskID | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| ADV-01 | manifest shrink | 84-check manifest | BLOCKER/3 | C019 | test_adv01_manifest_too_few_rejected | @ TES post C037–C039 |
| ADV-02 | duplicate ID | dup row | ValueError | C019 | test_adv02_duplicate_check_id_rejected | @ TES post C037–C039 |
| ADV-03 | Annex C drift | unknown ID | closed-world 3 | C019 | test_adv03_unknown_check_id_rejected | @ TES post C037–C039 |
| ADV-04 | DAG cycle | cycle edge | ValueError | C030 | test_adv04_cycle_rejected | @ TES post C037–C039 |
| ADV-05 | false FULL dry | dry_run FULL | INCOMPLETE 2 | C067 | test_adv05_dry_run_coverage_not_full | @ TES post C037–C039 |
| ADV-06 | FAST as FULL | FAST run | PARTIAL label | C004 | test_adv06_fast_never_full_label | @ TES post C037–C039 |
| ADV-07 | FAST floor skip | skip mandatory | INCOMPLETE 2 | C004 | test_adv07_fast_floor_incomplete_exit | @ TES post C037–C039 |
| ADV-08 | FREEZE no baseline | no bootstrap | BLOCKER 1 | C009 | test_adv08_freeze_without_baseline_blocked | @ TES post C037–C039 |
| ADV-09 | baseline path jail | bad path | ValueError | C001 | test_adv09_baseline_path_jail | @ TES post C037–C039 |
| ADV-10 | orphan dep | unknown dep | ValueError | C030 | test_adv10_orphan_dep_rejected | @ TES post C037–C039 |
| ADV-11 | mutating sandbox | mutating check | ValueError | C014 | test_adv11_sandbox_rejects_mutating | @ TES post C037–C039 |
| ADV-12 | evidence tamper | bit flip | verify fail | C026 | test_adv12_tampered_evidence_fails | @ TES post C037–C039 |
| ADV-13 | CHANGED-SINCE no sha | CLI | error 3 | C025 | test_adv13_changed_since_requires_since_sha_cli | @ TES post C037–C039 |
| ADV-14 | REGRESSED | severity bump | REGRESSED | C022 | test_adv14_regressed_on_severity_increase | @ TES post C037–C039 |
| ADV-15 | extra artifact | add file | verify fail | C026 | test_adv15_extra_artifact_fails_closed_verify | @ TES post C037–C039 |
| ADV-16 | path traversal | ../ entry | ValueError | C027 | test_adv16_path_traversal_rejected | @ TES post C037–C039 |
| ADV-17 | shell | shell=True | rejected | C014 | test_adv17_sandbox_never_shell_true | @ TES post C037–C039 |
| ADV-18 | digest swap | wrong digest | BLOCKER 1 | C028 | test_adv18_manifest_digest_mismatch_finding | @ TES post C037–C039 |
| ADV-19 | wrong SHA | fake HEAD | BLOCKER 1 | C012 | test_adv19_report_sha_matches_git_head | @ TES post C037–C039 |
| ADV-20 | tenant category | strip category | TEN enrich | C015 | test_adv20_tenant_enrich_restores_category | @ TES post C037–C039 |
| ADV-21 | VBN category | strip category | VBN enrich | C011 | test_adv21_vbn_enrich_restores_category | @ TES post C037–C039 |
| ADV-22 | non-atomic report | interrupt | no torn file | C013 | test_adv22_no_partial_report_without_status | @ TES post C037–C039 |
| ADV-23 | FRZ lie | all AUD-03 | BLOCKER 1 | C020 | test_adv23_frz_map_varies_by_check | @ TES post C037–C039 |
| ADV-24 | bootstrap accepted | bootstrap | no accepted | C002 | test_adv24_bootstrap_never_writes_accepted | @ TES post C037–C039 |
| ADV-25 | fake VBN NA | forged domains | VBN BLOCKER 1 | C043 | test_adv25_domain_gap_fails_full | @ TES post C037–C039 |
| ADV-26 | empty CHANGED-SINCE | since=HEAD | BLOCKER 1 | C025 | test_adv26_empty_changed_since_blocked | @ TES post C037–C039 |
| ADV-27 | bad mode | invalid mode | argparse 3 | C005 | test_adv27_invalid_mode_rejected | @ TES post C037–C039 |
| ADV-28 | CI gap | readiness CI section | fail | C054 | test_adv28_mechanical_readiness_ci_section | @ TES post C037–C039 |
| ADV-29 | non-cert bundle | non-cert run | build error | C068 | test_adv29_evidence_refused_when_non_certifying | @ TES post C037–C039 |
| ADV-30 | L3 false qual | exit 3 accepted | assert fail | C031 | test_adv30_l3_full_strict_exit_zero_only | @ TES post C037–C039 |
| ADV-31 | status mutation | PASS+findings | internal 3 | C044 | test_adv31_mutated_status_not_clean | @ TES post C037–C039 |


---

## 3. PL-07 — pytest taxonomy @ `34e0342`

```powershell
uv run pytest tests/qualification/ent_audit_x --collect-only -q
```

**Result @ `34e03425b2322f415a99a7cfd290fe0982bf6970`:** **184** tests collected · **85** `entrypoint_exists` · **0/85** semantic QUALIFIED @ TES.

---

## 4. OD-ENT-01 / ECAP (PL-08)

**Mandatory order:** B6 PASS → EVIDENCE_PUBLICATION_SHA → OD-ENT-01 → ACCEPTED_BASELINE_SHA.

Promotion @ HEAD: ladder pins + `operator_records/` + dossier SHA cross-ref. **Remaining (C065):** mandatory closed bundle on promote (AUD-01).

---

## 5. Baseline FSM (FCL-09)

Acyclic: NO_BASELINE → bootstrap candidate → L3 qual → evidence published → independent B6 → operator promote → normal FREEZE.

---

## 6. Mechanical readiness (PL-09 / C066)

@ HEAD script returns `technical_ready_for_independent_b6_mechanical: false` (expected). Post-§7: all 15 inputs in §6 table must parse real artifacts.

---

## 7. Final qualification protocol (after R0–R4 integration)

1. Commit + push implementation.  
2. Clean checkout @ pushed SHA.  
3. Pin **TEST_EXECUTION_SHA** (append-only ladder).  
4. L1 + L2 pytest @ TES.  
5. Certifying **FULL 85/85** (C031, C074).  
6. **First FREEZE candidate** (C032).  
7. **TEN13 + VBN9 + D01..D31**.  
8. **ADV31** (C037–C039).  
9. **BAT + SH** (C063).  
10. **PR FAST + nightly FULL** CI evidence in dossier.  
11. Evidence integrity + replay (C026, C068, ADV tamper).  
12. Pin **EVIDENCE_PUBLICATION_SHA**.  
13. Mechanical readiness (C066) true only if all gates proven.  
14. Independent **Protocol 03 B6** (C058 handoff).  
15. Operator **OD-ENT-01** promote (C065) — **after** B6 only.

Requalify from step 3 on any semantic change after pin.

---

## 8. R0–R5 batches

| Batch | TaskIDs |
| --- | --- |
| R0 | C001–C008, C065–C069, C073 |
| R1 | C009–C018, C071, C022, C023 |
| R2 | C019–C030, C072 |
| R3 | C031–C044, C074 |
| R4 | C045–C047, C059–C063, C070, C007, C046 |
| R5 | C048–C058, C054–C056, C064, C066, C075, C053 |

**R5 doc vs evidence:** C053/C048/C075 prepare structure; **do not** insert fictional TES/EVIDENCE/B6 SHAs before §7.

**C057:** agent-authored OD authority matrix document (operator decisions remain external gates).

---

## 9. Implementation inventory (Tier-0 @ HEAD)

| Component | Path | Contract | Behavior @ HEAD | Tests | Gap/Task |
| --- | --- | --- | --- | --- | --- |
| Runner | `runner.py` | certifying modes | dry_run → non-certifying | unit/integration | C067,C009,C031 |
| Baseline | `baseline.py` | FSM + promote | OD verify partial | test_baseline_promotion | C065,C003 |
| Evidence | `evidence.py` | closed bundle | certifying rules partial | test_certifying_and_bundle | C068,C006 |
| Validators | `validators.py` | schema verify | shallow | test_json_schema_report | C069,C026 |
| Sandbox | `sandbox.py` | allowlist exec | partial isolation | test_sandbox_path_traversal | C014-C017 |
| VBN | `vbn.py` | D01..D31 rollup | script invoke | unit partial | C008-C011 |
| Checks | `checks/executor.py` | manifest exec | wiring/skip pending | acceptance exists | C033-C036 |
| CLI | `run_enterprise_audit.py` | exit 0-3 | modes | integration | C005,C059 |
| CI | `enterprise-audit.yml` | PR FAST | real FAST + pytest | workflow | C059,C046 |
| Launchers | `audit.bat`/`audit.sh` | one-command | unknown arg bug | partial | C060-C063 |
| Readiness | `mechanical_readiness_check.py` | fail-closed | stubs false | test_mechanical_readiness | C066 |
| Qual tree | `tests/qualification/ent_audit_x/` | L1/L2/L3/acc | 184 collect | pytest | C031-C041 |

---

## 14. Task DAG (executable)

**Topological order:**  
C004,C005 → C001,C008,C019,C030 → C067,C069,C073,C006 → C065→C003 → C068 → C026,C027 → C008→C011→C018 → C071,C022,C023 → C024→C025→C029→C072 → C033 → C034,C035,C036 → C037→C038→C039 → C031,C032,C074 → C045,C059,C007,C046,C060,C061,C063,C070 → **§7 qual** → C049,C050,C051 → C053,C058,C066 → C048,C055,C056,C057,C064,C075.

**Shared-file serial owners:** `baseline.py` (R0); `runner.py` (R0→R1); `modes.py` (R0/R1); `evidence.py`/`validators.py` (R0/R2); `vbn.py` (R1); `changed_since.py` (R2); readiness script verify-only until post-qual.

**Safe parallelism:** C034|C035|C036 after C033; C037–C039 parallel; C055|C057|C056 doc after C041.

---

## 16. Mechanical completeness @ plan publication

| Check | Result |
| --- | --- |
| AUD-01..44 | **44/44** Matrix B |
| C001..C075 exactly once | **75/75** Matrix C |
| GAP-RC + GAP-B + GAP-COV | **33+6+85** Matrix D |
| CheckIDs | **85/85** Matrix E |
| ADV-01..31 | **31/31** Matrix F |
| FRZ-AUD-01..10 | Matrix A.2 |
| Dual-axis sums | **PASS** ({'PARTIAL': 21, 'IMPLEMENTED': 4, 'NOT_STARTED': 50}, {'NOT_QUALIFIED': 73, 'QUALIFIED': 2}) |
| DAG acyclic | **PASS** §14 |
| No fabricated qual SHA | **PASS** |
| Operator gates separated | **PASS** |
| C057 = agent doc | **PASS** |

---

## 13. One-pass agent instruction

1. R0→R5 in DAG order.  
2. Do not re-implement C004,C005.  
3. §7 only after R4 push.  
4. Stop at mechanical readiness; external B6 + operator closure.

---

## Appendix B — C065..C075 (NETNEW specs)

See [corrective plan](./ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md) + Matrix C columns.

---

## R5 annex / ladder / dossier tasks (no fictional SHAs)

| Artifact | Task | Action @ implementation wave |
| --- | --- | --- |
| [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md) | C048,C075 | Append `IMPLEMENTATION_SHA` row for each implementation commit; pin TES only in §7 step 3 |
| [ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md](./ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md) | C053,C075 | Fill TES/EVID/B6 fields only after qual + B6 |
| [ANNEX_F_GAP_REGISTER.md](./ANNEX_F_GAP_REGISTER.md) | C055 | Status columns @ HEAD; preserve historical GAP-C closed rows |
| [ANNEX_H_WBS.md](./ANNEX_H_WBS.md) | C055,C075 | Align 75 TaskID rows to Matrix C |
| [ANNEX_G2_CONTROL_GATE_MAPPING.md](./ANNEX_G2_CONTROL_GATE_MAPPING.md) | C020,C055 | Crosswalk 63 ENT-A + TEN/VBN via Annex C (AUD-36) |

---

*Execution SSOT published — implementation baseline `34e03425b2322f415a99a7cfd290fe0982bf6970`; agent wave R0–R5 follows §8 DAG (no qual §7 until R4 lands).*
