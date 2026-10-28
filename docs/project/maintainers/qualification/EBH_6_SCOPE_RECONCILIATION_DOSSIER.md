# EBH-6 — Scope Reconciliation Dossier (SCOPE-01..03)

| Field | Value |
| --- | --- |
| **Status** | **EBH-6 — P03 INTEGRITY CORRECTED / PARTIAL — EXACT BLOCKERS REMAIN** |
| **Independent auditor disposition** | **23 × PENDING** — implementer must not set PASS |
| **EXPECTED_START_HEAD** | `1858eafb84cfcc6c9197556bb24804e6bce864d8` |
| **IMPLEMENTATION_SHA** | `bccaf6ed24ca443ddf8d8ad6f4463e709b0f7289` |
| **TEST_EXECUTION_SHA** | `bccaf6ed24ca443ddf8d8ad6f4463e709b0f7289` |
| **CERTIFICATION_HANDOFF_SHA** | `d7b6442887941b6941a291b0efcf1271b9aee1de` |

Mechanical SSOT: [`tests/qualification/ebh_6/_l3_aj_group_reconciliation.py`](../../../tests/qualification/ebh_6/_l3_aj_group_reconciliation.py), [`_l3_reachability_evidence.py`](../../../tests/qualification/ebh_6/_l3_reachability_evidence.py), [`_family_l9_evidence.py`](../../../tests/qualification/ebh_6/_family_l9_evidence.py).

Roadmap authority: [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) § EBH-6 / §3.0.8. VBN: [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md).

## SCOPE-01 — A–J grouped reconciliation

352 prior `EVIDENCE_REQUIRED` cells reconciled via **one row per (layer, applicable class)** — not 352 micro-iterations.

| **PASS_CANDIDATE** (udowodnione — bezpośredni gate) | 3 |
| **PARTIAL_CANDIDATE** (częściowo) | 52 |
| **EVIDENCE_REQUIRED** (brakujący dowód @ HEAD) | 342 |
| **FINDING** | 2 |
| **N/A — WITH EVIDENCE** | 81 |

**Positive proofs (group bundles):** register `unclassified=0`, `prohibited unresolved=0`; CONFIG-X 54/54 mapping; tier boundary gate; sanctioned roots; layer gate tests from `_layer_evidence.layer_gate_test_ids`; behavioral gates cited per mechanism `L3_gate=`.

**Adversarial negatives (unchanged):** `EBH6-DEBT-Q2-D2` (class F FINDING); `EBH6-DEBT-BND-COMPAT-R3` (class H FINDING, lab-only).

**Prohibited unresolved @ frozen scope:** 0.

## SCOPE-02 — Families CMP / PRD / REG / DOC

| family | disposition | closure |
| --- | --- | --- |
| **CMP** | SCOPED CANDIDATE | COMPAT-X historical closure; EVOL consumer impact OUT OF SCOPE (Path B) |
| **PRD** | SCOPED CANDIDATE | Live PROD-Q vendor slice OUT OF SCOPE; architecture census via register |
| **REG** | SCOPED CANDIDATE | EBH-6 qual regression bundle (`mechanical_exit` + register + L3/family integrity) — **not** QUAL-X / global FRZ-REG |
| **DOC** | SCOPED CANDIDATE | Mandatory docs + this dossier content gate (`test_ebh6_scope_dossier_content_controls`) — **not** global FRZ-DOC PASS |

## 16 layers — register status

All `EBH6-LAYER-*`: **`L9_EVIDENCE_SUBMITTED`** with ≥3 L3 mechanisms each (`_l3_reachability_evidence.py`).

## 23 families — implementer proposals

| disposition | count |
| --- | ---: |
| PASS_CANDIDATE | 2 (EXE, GOV) |
| SCOPED CANDIDATE | 21 |
| EVIDENCE REQUIRED | 0 |

See [`EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md`](EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md) + `_family_l9_evidence.l9_evidence_disposition`.

## CONFIG-X reconciliation

@ HEAD: **54/54** concerns mapped — gate `test_ebh6_config_x_all_concerns_mapped_at_head`; historical PASS @ `fac9a43f…` preserved (no FRZ-CFG rewrite).

## Unresolved mandatory blockers (certification)

| ID | Status |
| --- | --- |
| **OD-EBH6-01** | OPEN (no production wave) |
| **OD-EBH6-08** | OPEN (parent closure) |
| **L9 Protocol 03** | Independent re-audit required — **not** implementer PASS |
| **51 EBH-5 NYQ** | Deferred EBH-7 |
| **DUP-X** | Not started |
| **8 × EBH6-BLK-L9-*** | Implementer evidence submitted — independent PENDING |

## Explicit non-claims

- No **Protocol 03 PASS**, no **EBH-6 CLOSED**, no global **FRZ PASS**, no agent **AUDIT PASS**.
- Group PASS_CANDIDATE ≠ independent acceptance.

## Verification (@ TEST_EXECUTION_SHA)

```powershell
uv run pytest tests/qualification/ebh_6/ tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py -q
```

Log: `.tmp/session/ebh-6-scope-reconciliation/pytest.log`
