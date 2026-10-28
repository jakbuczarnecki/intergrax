# EBH-6 — Protocol 03 Rejection Remediation (B-L9-5)

| Field | Value |
| --- | --- |
| **Disposition** | **EBH-6 — PROTOCOL 03 REJECTION REMEDIATED / READY FOR RE-AUDIT** |
| **Rejection root cause** | Mechanical catalog completeness ≠ production-path invariant evidence (`paths[:3]`, template L3, catalog-only family gates) |
| **IMPLEMENTATION_SHA** | `475c20289a1ef0420461a1ba748c792a10ea3e99` |
| **TEST_EXECUTION_SHA** | `475c20289a1ef0420461a1ba748c792a10ea3e99` |
| **PUBLICATION_SHA** | *(set on doc commit)* |
| **Historical L9 wave** | `bc25e27b14b7172ef79951765e76d19578d9beb3` / `ce148559539583f4569d7a43c245e6c830e60a06` — **HISTORICAL** |

## R1 — L3 reachability (16 layers)

| Deliverable | Location |
| --- | --- |
| Per-mechanism records (≥3/layer) | [`tests/qualification/ebh_6/_l3_reachability_evidence.py`](../../../tests/qualification/ebh_6/_l3_reachability_evidence.py) |
| Layer inventory + L3 index | [`tests/qualification/ebh_6/_layer_evidence.py`](../../../tests/qualification/ebh_6/_layer_evidence.py) |
| Register status | **`L9_EVIDENCE_SUBMITTED`** — not independent RESOLVED |

## R2 — Architecture families (23 + 8 blockers)

| Deliverable | Location |
| --- | --- |
| Family L9 bundles | [`tests/qualification/ebh_6/_family_l9_evidence.py`](../../../tests/qualification/ebh_6/_family_l9_evidence.py) |
| Catalog overlay @ HEAD | [`tests/qualification/ebh_6/_family_catalog.py`](../../../tests/qualification/ebh_6/_family_catalog.py) |
| Matrix narrative | [EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md](EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md) |

## R3 — Mechanical integrity

| Test module | Purpose |
| --- | --- |
| [`test_ebh_6_l3_and_family_evidence_integrity.py`](../../../tests/qualification/ebh_6/test_ebh_6_l3_and_family_evidence_integrity.py) | L3 chains, anti-`paths[:3]`, blocker gates ≠ catalog-only |
| [`test_ebh_6_hardcoding_register_integrity.py`](../../../tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py) | Layer rows **`L9_EVIDENCE_SUBMITTED`** + ≥3 mechanism IDs |

## R4 — Verification @ IMPLEMENTATION_SHA

```powershell
uv run pytest tests/qualification/ebh_6/ tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py -q
```

Log: `.tmp/session/ebh-6-p03-remediation/pytest.log` — **52 passed**.

## Explicit non-claims

- No Protocol 03 PASS, no **OD-EBH6-08**, no EBH-6 CLOSED, no global FRZ PASS
- **INDEPENDENT_AUDITOR_DISPOSITION** remains **23 × PENDING**
- **OD-EBH6-01** OPEN — production wave not operator-open
- **PROHIBITED** unresolved in frozen scope: **0**

## Remaining blockers (outside remediated evidence)

| ID | Status |
| --- | --- |
| OD-EBH6-01 | OPEN |
| OD-EBH6-08 | OPEN |
| L9 Protocol 03 re-audit | **READY FOR RE-AUDIT** @ new CERTIFICATION_HANDOFF_SHA |
| 51 EBH-5 NYQ rows | Deferred (EBH-7) |
| DUP-X | Not started |
