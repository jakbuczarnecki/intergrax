# EBH-6 — Evidence Freshness Ledger

| Field | Value |
| --- | --- |
| **IMPLEMENTATION_SHA** | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` |
| **QUALIFICATION_SHA** | Same as TEST_EXECUTION_SHA when pytest recorded @ run time |
| **EXPECTED_START_HEAD** | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` |
| **TEST_EXECUTION_SHA** | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` |
| **CERTIFICATION_HANDOFF_SHA** | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` |

## SHA ladder

| Label | SHA | Freshness |
| --- | --- | --- |
| CURRENT HEAD / IMPLEMENTATION | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` | **CURRENT** |
| EBH-6 publication tree (historical) | `50d3b7785c7e13dbc6708143e7484a2eeea5eaca` | **HISTORICAL** — prior P0–P6 artifact commit |
| PRE-PUBLICATION local batch (historical) | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` | **HISTORICAL** — superseded; do not cite as HEAD |
| EBH-5 independent audit | `33ecf19cc020b112d9e586c87b49e819202f5d6d` | **HISTORICAL — scoped PASS** |
| EBH-5 wave baseline | `83a376b3e1b312e2e4f88da548c99c328f3589f7` | **HISTORICAL** |
| CONFIG-X closed-world PASS | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` | **HISTORICAL — preserved** |
| CTRL-X independent audit | `318a4d835534ef5c48575f4649551c957dacd8ea` | **HISTORICAL — CLOSED** |

## Prior stage evidence disposition

| Owner stage | Artifact class | EBH-6 treatment |
| --- | --- | --- |
| EBH-5 | VBN inventory, matrix, reachability, behavioral proofs | **Replay @ HEAD** — no global FRZ PASS; parent **CLOSED** @ OD-EBH5-08 |
| CONFIG-X | FRZ-CFG-01..08 @ fac9a43f | **Reconciliation only** — see neutrality audit |
| CTRL-X | FRZ-CTL-01..12 | **Drift check** — matrix row REPLAY PASS |
| TRACE-X | Sanctioned composition registry | **CURRENT** — wired in closed-world discovery |
| STATE-X / TENANT-X / PROD-Q live | Various | **HISTORICAL or OUT OF SCOPE** per dossier |

## EBH-6 new evidence (CURRENT)

| Artifact | Path |
| --- | --- |
| Pre-cert findings | [EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md) |
| Findings register | [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) + `_findings_register.py` |
| Neutrality rollup | [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md) |
| Family matrix | [EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md](EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md) |
| Qualification gates | `tests/qualification/ebh_6/` |

## Blockers recorded

- **OD-EBH6-01** OPEN — implementation wave not operator-open
- **OD-EBH6-02** OPEN — plan audit
- **L9 Protocol 03** OPEN
