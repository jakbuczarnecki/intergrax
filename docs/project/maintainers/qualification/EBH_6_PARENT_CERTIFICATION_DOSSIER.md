# EBH-6 — Parent Certification Dossier

| Field | Value |
| --- | --- |
| **Stage** | EBH-6 — Final Architecture Recertification |
| **Status** | **READY FOR AUDIT** — **not** `CLOSED — INDEPENDENTLY ACCEPTED` |
| **START_HEAD** (local batch) | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` |
| **STAGE_BASELINE_SHA** (operator / origin) | `d3953f2dee5ca2ab56954e2bf64e4b4f5af828c6` — on **origin/development**; local branch **behind** (see [EBH_6_GIT_RECONCILIATION_REPORT.md](EBH_6_GIT_RECONCILIATION_REPORT.md)) |
| **IMPLEMENTATION_SHA** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` |
| **TEST_EXECUTION_SHA** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` — **PRE-PUBLICATION** |
| **QUALIFICATION_SHA** | Same as TEST_EXECUTION_SHA when pytest @ run time |
| **PUBLICATION_SHA** | **TBD** — untracked artifacts; no agent commit |

## Latest pytest (@ TEST_EXECUTION_SHA)

| Run | Result |
| --- | --- |
| `uv run pytest tests/qualification/ebh_6/ -q` | **23 passed** (2026-10-28, PRE-PUBLICATION) |
| Log | `.tmp/session/ebh-6-publication-reconcile/pytest.log` |

## Mandatory blockers

- **OD-EBH6-01** OPEN — wave / `production delta` (roadmap §3.0.8)
- **OD-EBH6-02**, **OD-EBH6-08** OPEN
- **L9 / Protocol 03** — no pass record
- Git — local HEAD behind operator **STAGE_BASELINE_SHA**
- Qualification tree **uncommitted** — operator publication authorization required

## Reconcile pass artifacts

- [EBH_6_GIT_RECONCILIATION_REPORT.md](EBH_6_GIT_RECONCILIATION_REPORT.md)
- [EBH_6_P0_P6_RECONCILIATION_AND_CERTIFICATION_HANDOFF.md](EBH_6_P0_P6_RECONCILIATION_AND_CERTIFICATION_HANDOFF.md)

## Artifact index

| Artifact | Location |
| --- | --- |
| Operator decisions | [EBH_6_OPERATOR_DECISION_REGISTER.md](EBH_6_OPERATOR_DECISION_REGISTER.md) |
| Surface inventory | [EBH_6_ENTERPRISE_SURFACE_INVENTORY.md](EBH_6_ENTERPRISE_SURFACE_INVENTORY.md) |
| Freshness ledger | [EBH_6_EVIDENCE_FRESHNESS_LEDGER.md](EBH_6_EVIDENCE_FRESHNESS_LEDGER.md) |
| Family matrix | [EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md](EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md) |
| Hardcoding register | [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) |
| Neutrality audit | [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md) |
| L9 handoff | [EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md) |
| Readiness | [EBH_6_STAGE_CERTIFICATION_READINESS.md](EBH_6_STAGE_CERTIFICATION_READINESS.md) |
| Qualification SSOT | [`tests/qualification/ebh_6/`](../../../tests/qualification/ebh_6/) |

## EBH-5 reuse (by reference)

| Artifact | EBH-5 path |
| --- | --- |
| Inventory | [EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md](EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md) |
| Matrix | [EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md](EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md) |
| Protocol 03 PASS (scoped) | [EBH_5_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md](EBH_5_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md) |

## FRZ traceability

Contributions toward current-HEAD evidence for applicable families — **does not** promote global FRZ PASS. See family matrix + findings register for SYS-INV / FRZ linkage per row.

## Known limitations (auditor ack required)

- **OD-EBH5-08** OPEN — EBH-5 parent closure not published
- **OD-EBH6-01** OPEN — wave open phrase not recorded
- **51** EBH-5 NYQ rows
- **PROD-Q** live slice out of scope
- **DUP-X** deferred

## Verification command

```powershell
uv run pytest tests/qualification/ebh_6/ -q
```

## Forbidden claims

- Global **FRZ PASS**
- **EBH-6 — CLOSED**
- Agent-issued **INDEPENDENT AUDIT PASS**
