# EBH-6 — Parent Certification Dossier

| Field | Value |
| --- | --- |
| **Stage** | EBH-6 — Final Architecture Recertification |
| **Status** | **PARTIAL / EXACT BLOCKERS (OD-EBH6-Q2D2 OPEN)** — implementer evidence complete; **not** independent P03 PASS |
| **CLOSURE_BASELINE_SHA** | `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7` |
| **IMPLEMENTATION_SHA** | **CERTIFICATION_HANDOFF_SHA** |
| **TEST_EXECUTION_SHA** | **CERTIFICATION_HANDOFF_SHA** |
| **CERTIFICATION_HANDOFF_SHA** | @ final enterprise closure wave (see ledger) |

## Latest pytest (@ TEST_EXECUTION_SHA)

| Run | Result |
| --- | --- |
| Consolidated L3 (final closure E09) | @ **TEST_EXECUTION_SHA** — `.tmp/session/ebh-6-final-closure/pytest.log` |

## Mandatory blockers

- **OD-EBH6-01** OPEN — wave / `production delta` (roadmap §3.0.8)
- **OD-EBH6-08** OPEN — parent closure forbidden until L9 PASS
- **L9 / Protocol 03** — **NOT EXECUTED** — execution package @ [EBH_6_PROTOCOL_03_EXECUTION_PACKAGE.md](EBH_6_PROTOCOL_03_EXECUTION_PACKAGE.md)
- **OD-EBH6-02** **APPROVED 2026-10-28** — plan execution only

## L9 evidence (implementer)

- **16/16** layers: B5 A–J class gates + L3 annex — **`CANDIDATE-RESOLVED`**
- **23/23** families: `_family_l9_evidence.py` + catalog overlay; **independent disposition 23 × PENDING**

## Artifact index

| Artifact | Location |
| --- | --- |
| L9 plan | [EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md](EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md) |
| **Closure plan (B1–B7)** | [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) @ `0489054c…` |
| P03 remediation | [EBH_6_PROTOCOL_03_REJECTION_REMEDIATION.md](EBH_6_PROTOCOL_03_REJECTION_REMEDIATION.md) |
| Scope reconciliation | [EBH_6_SCOPE_RECONCILIATION_DOSSIER.md](EBH_6_SCOPE_RECONCILIATION_DOSSIER.md) |
| A–J group SSOT | [`tests/qualification/ebh_6/_l3_aj_group_reconciliation.py`](../../../tests/qualification/ebh_6/_l3_aj_group_reconciliation.py) |
| L3 annex | [`tests/qualification/ebh_6/_l3_reachability_evidence.py`](../../../tests/qualification/ebh_6/_l3_reachability_evidence.py) |
| Pre-cert findings | [EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md) |
| Operator decisions | [EBH_6_OPERATOR_DECISION_REGISTER.md](EBH_6_OPERATOR_DECISION_REGISTER.md) |
| Freshness ledger | [EBH_6_EVIDENCE_FRESHNESS_LEDGER.md](EBH_6_EVIDENCE_FRESHNESS_LEDGER.md) |
| Family matrix | [EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md](EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md) |
| Hardcoding register | [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) |
| L9 handoff | [EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md) |
| Readiness | [EBH_6_STAGE_CERTIFICATION_READINESS.md](EBH_6_STAGE_CERTIFICATION_READINESS.md) |
| Qualification SSOT | [`tests/qualification/ebh_6/`](../../../tests/qualification/ebh_6/) |

## Known limitations (auditor ack required)

- **OD-EBH6-01** OPEN
- **51** EBH-5 NYQ rows (EBH-7)
- **PROD-Q** live slice out of scope
- **DUP-X** deferred
- Family **L9 blockers:** EBH6-BLK-L9-OWN-01, EXE-01, GOV-01, REC-01, SEC-01, REL-01, HRN-01, TEN-01

## Verification command (consolidated)

```powershell
uv run pytest tests/qualification/ebh_6/ tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py -q
```

## Forbidden claims

- Global **FRZ PASS**
- **EBH-6 — CLOSED**
- Agent-issued **INDEPENDENT AUDIT PASS**
