# EBH-6 — Parent Certification Dossier

| Field | Value |
| --- | --- |
| **Stage** | EBH-6 — Final Architecture Recertification |
| **Status** | **EBH-6 — FINAL L9 EVIDENCE READY FOR INDEPENDENT CERTIFICATION** — **not** `CLOSED — INDEPENDENTLY ACCEPTED` |
| **IMPLEMENTATION_SHA** | `4736e4e4a00aff44afea9a28f91d188631ae43f1` |
| **TEST_EXECUTION_SHA** | *(set after consolidated pytest — see ledger)* |
| **CERTIFICATION_HANDOFF_SHA** | *(aligned to IMPLEMENTATION_SHA)* |
| **EXPECTED_START_HEAD** | `4736e4e4a00aff44afea9a28f91d188631ae43f1` |
| **PUBLICATION_SHA (P0–P6 historical)** | `50d3b7785c7e13dbc6708143e7484a2eeea5eaca` |

## Latest pytest (@ TEST_EXECUTION_SHA)

| Run | Result |
| --- | --- |
| Consolidated L4 bundle (plan L9-E) | `.tmp/session/ebh-6-l9/pytest.log` |

## Mandatory blockers

- **OD-EBH6-01** OPEN — wave / `production delta` (roadmap §3.0.8)
- **OD-EBH6-08** OPEN — parent closure forbidden until L9 PASS
- **L9 / Protocol 03** — no pass record (**B-L9-5 not executed**)
- **OD-EBH6-02** **APPROVED 2026-10-28** — plan execution only

## L9 evidence (implementer)

- **16/16** layers: `_layer_evidence.py` + RESOLVED `EBH6-LAYER-*` register rows
- **23/23** families: `implementer_proposed_disposition` populated; **independent disposition 23 × PENDING**

## Artifact index

| Artifact | Location |
| --- | --- |
| L9 plan | [EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md](EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md) |
| Layer annex | [`tests/qualification/ebh_6/_layer_evidence.py`](../../../tests/qualification/ebh_6/_layer_evidence.py) |
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
