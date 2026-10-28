# EBH-6 — Parent Certification Dossier

| Field | Value |
| --- | --- |
| **Stage** | EBH-6 — Final Architecture Recertification |
| **Status** | **EBH-6 B2 COMPLETE — PARTIAL / EXACT BLOCKERS** |
| **Closure plan** | [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) @ `0489054c5f80ee392211fb33a39f2736fa2fdfa0` |
| **B2 EXPECTED_START_HEAD** | `83b0a26186b0c11430c916842d7d813554286921` |
| **IMPLEMENTATION_SHA** | *(set @ B2 qual commit)* |
| **TEST_EXECUTION_SHA** | *(set @ B2 qual commit)* |
| **CERTIFICATION_HANDOFF_SHA** | `3f0a25439270b173c63989d2fd948c82a8cff6af` — **pre-B1** |
| **PUBLICATION_SHA (P0–P6 historical)** | `50d3b7785c7e13dbc6708143e7484a2eeea5eaca` |

## Latest pytest (@ TEST_EXECUTION_SHA)

| Run | Result |
| --- | --- |
| Consolidated L4 bundle (scope reconciliation) | **57 passed** — `.tmp/session/ebh-6-scope-reconciliation/pytest.log` |

## Mandatory blockers

- **OD-EBH6-01** OPEN — wave / `production delta` (roadmap §3.0.8)
- **OD-EBH6-08** OPEN — parent closure forbidden until L9 PASS
- **L9 / Protocol 03** — **READY FOR RE-AUDIT** @ CERTIFICATION_HANDOFF_SHA (prior reject remediated — not PASS)
- **OD-EBH6-02** **APPROVED 2026-10-28** — plan execution only

## L9 evidence (implementer)

- **16/16** layers: B2 closed-world census (`_closed_world_discovery.py`) + **49** L3 rows — **`L9_EVIDENCE_SUBMITTED`**
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
