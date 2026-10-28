# EBH-6 — Evidence Freshness Ledger

| Field | Value |
| --- | --- |
| **IMPLEMENTATION_SHA** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` |
| **QUALIFICATION_SHA** | Same as IMPLEMENTATION_SHA when pytest recorded @ run time |
| **EXPECTED_START_HEAD / STAGE_BASELINE** | `d3953f2dee5ca2ab56954e2bf64e4b4f5af828c6` — **on origin/development** after fetch; **local HEAD behind** |
| **TEST_EXECUTION_SHA** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` (PRE-PUBLICATION pytest) |

## SHA ladder (P0)

| Label | SHA | Freshness |
| --- | --- | --- |
| STAGE_BASELINE_SHA (operator EXPECTED_START) | `d3953f2…` | **origin/development** — local must fast-forward before publication SHA |
| IMPLEMENTATION_SHA / HEAD | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` | **CURRENT** |
| EBH-5 independent audit | `33ecf19cc020b112d9e586c87b49e819202f5d6d` | **HISTORICAL — scoped PASS** |
| EBH-5 wave baseline | `83a376b3e1b312e2e4f88da548c99c328f3589f7` | **HISTORICAL** |
| CONFIG-X closed-world PASS | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` | **HISTORICAL — preserved** |
| CTRL-X independent audit | `318a4d835534ef5c48575f4649551c957dacd8ea` | **HISTORICAL — CLOSED** |

## Prior stage evidence disposition

| Owner stage | Artifact class | EBH-6 treatment |
| --- | --- | --- |
| EBH-5 | VBN inventory, matrix, reachability, behavioral proofs | **Replay @ HEAD** — no global FRZ PASS |
| CONFIG-X | FRZ-CFG-01..08 @ fac9a43f | **Reconciliation only** — see neutrality audit |
| CTRL-X | FRZ-CTL-01..12 | **Drift check** — matrix row REPLAY PASS |
| TRACE-X | Sanctioned composition registry | **CURRENT** — wired in closed-world discovery |
| STATE-X / TENANT-X / PROD-Q live | Various | **HISTORICAL or OUT OF SCOPE** per dossier |

## EBH-6 new evidence (CURRENT)

| Artifact | Path |
| --- | --- |
| Findings register | [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) + `_findings_register.py` |
| Neutrality rollup | [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md) |
| Family matrix | [EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md](EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md) |
| Qualification gates | `tests/qualification/ebh_6/` |

## Blockers recorded

- **OD-EBH5-08** OPEN — EBH-5 parent not formally CLOSED ([`EBH_5_OD_EBH5_08_CLOSURE_SIGNATURE_RECORD.md`](EBH_5_OD_EBH5_08_CLOSURE_SIGNATURE_RECORD.md))
- **OD-EBH6-01** OPEN — implementation wave not operator-open
