# EBH-6 — Current-HEAD Vendor Neutrality Audit (VBN §2.7 rollup)

| Field | Value |
| --- | --- |
| **Purpose** | Current-HEAD rollup per [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2.7 |
| **Closure plan** | [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) @ `0489054c5f80ee392211fb33a39f2736fa2fdfa0` |
| **B2 EXPECTED_START_HEAD** | `83b0a26186b0c11430c916842d7d813554286921` — **HISTORICAL** |
| **P03 EXPECTED_START_HEAD** | `24cd97299cb412c610b19c86b29f99ace707baff` |
| **IMPLEMENTATION_SHA** | `b313835d7cf035b503eea56958b92eebfb43f997` |
| **TEST_EXECUTION_SHA** | `b313835d7cf035b503eea56958b92eebfb43f997` |
| **CERTIFICATION_HANDOFF_SHA** | `b313835d7cf035b503eea56958b92eebfb43f997` |
| **Historical P03 handoff (EV=0 mask)** | `129eee82dff33fd4d815643387a10fa2d56058bf` — **HISTORICAL / REJECTED** |
| **CONFIG-X historical PASS** | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` |
| **Historical IMPLEMENTATION (pre-cert)** | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` — **HISTORICAL** |
| **P03 integrity IMPLEMENTATION** | `1f3fb2ae3b8809dd59eb84f0a8d7d8f471aedbd9` — **HISTORICAL** |
| **Disposition** | **P03 INTEGRITY REMEDIATION — PARTIAL / EXACT EVIDENCE BLOCKERS** (`EV_GROUPS=109`; Q2-D2 FINDING / OD-EBH6-Q2D2 OPEN) |

## Mandatory audit layers (roadmap §3.0.8)

16 layers inventoried @ HEAD via [`tests/qualification/ebh_6/_closed_world_discovery.py`](../../../tests/qualification/ebh_6/_closed_world_discovery.py) and [`EBH_6_ENTERPRISE_SURFACE_INVENTORY.md`](EBH_6_ENTERPRISE_SURFACE_INVENTORY.md).

Register rows **`EBH6-LAYER-01..16`**: status **`L9_EVIDENCE_SUBMITTED`** (≥1 mandatory A–J gap per layer — not CANDIDATE-RESOLVED).

L3 per-mechanism annex: [`_l3_reachability_evidence.py`](../../../tests/qualification/ebh_6/_l3_reachability_evidence.py) (**49** mechanisms — B2 graph bootstrap + hosting path correction).

Grouped A–J adjudication: [`_l3_aj_group_reconciliation.py`](../../../tests/qualification/ebh_6/_l3_aj_group_reconciliation.py) — **no unconditional PASS promotion (P03-01)**.

## A–J obligation groups (@ IMPLEMENTATION_SHA)

Mechanical census @ **P03 integrity remediation** (`tests/qualification/ebh_6/` — `.tmp/session/ebh-6-evidence-integrity/pytest.log`):

| Outcome | Groups (133 applicable pairs) |
| --- | ---: |
| EVIDENCE_REQUIRED | **109** |
| PARTIAL_CANDIDATE | **22** |
| PASS_CANDIDATE | **1** |
| FINDING | **1** (`semantic_core` × F) |

## CORE_HARDCODING_FINDINGS_REGISTER (@ HEAD)

| Metric | Value |
| --- | --- |
| Total rows | **57** |
| unclassified (inventory-style) | **0** (`test_ebh6_findings_unclassified_zero`) |
| PROHIBITED unresolved (frozen scope) | **0** (`test_ebh6_prohibited_unresolved_zero`) |
| Layer L9 rows | **16 × L9_EVIDENCE_SUBMITTED** |
| Tracked findings | Q2-D2 **RATIFIED**; BND-COMPAT-R3 **RATIFIED** (lab-only) |

SSOT: [`_findings_register.py`](../../../tests/qualification/ebh_6/_findings_register.py) · narrative: [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md).

## CONFIG-X reconciliation

| Metric | Value |
| --- | --- |
| Concerns | **54** |
| Mapped @ HEAD | **54/54** |
| Gate | `test_ebh6_config_x_all_concerns_mapped_at_head` |
| Historical FRZ-CFG PASS | **Preserved** @ `fac9a43f…` — no PASS rewrite |
| RPL-G active CONFIG-X blockers @ HEAD | **Empty** (EBH-5 gate) |

## EBH-5 / replaceability limitations

| Item | Status |
| --- | --- |
| Vendor inventory replay | **unclassified=0** @ EBH-5 mechanical gates |
| Cross-layer matrix | **51 rows NOT YET QUALIFIED** — EBH-7 scope |
| Global FRZ-RPL PASS | **Not claimed** |

## Architecture families (23)

SSOT: [`_family_l9_evidence.py`](../../../tests/qualification/ebh_6/_family_l9_evidence.py). **Independent disposition: 23 × PENDING.**

## Explicit scope exclusions

- **PROD-Q** live vendor slice — Path B OUT OF SCOPE
- **DUP-X** — not started
- **OD-EBH6-01** production wave — OPEN (no production delta in this batch)

## Mandatory blockers (certification — unchanged by implementer)

| ID | Status |
| --- | --- |
| **OD-EBH6-01** | OPEN |
| **OD-EBH6-08** | OPEN |
| **L9 Protocol 03** | Independent audit **PENDING** — candidate handoff only |
| **EBH6-P03-AJ-* ** | Per (layer, class) `EVIDENCE_REQUIRED` groups — see group table |
| **8 × EBH6-BLK-L9-*** | Implementer gates submitted — auditor PENDING |
| **51 NYQ** | Deferred EBH-7 |

## Candidate disposition (Protocol 03)

Implementer submits **PASS_CANDIDATE / PARTIAL_CANDIDATE / EVIDENCE_REQUIRED / FINDING / N/A** matrix above for auditor replay @ **CERTIFICATION_HANDOFF_SHA**. **No agent-issued AUDIT PASS**, no EBH-6 CLOSED, no global FRZ PASS.

Consolidated dossier: [EBH_6_SCOPE_RECONCILIATION_DOSSIER.md](EBH_6_SCOPE_RECONCILIATION_DOSSIER.md) · [EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md](EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md) · [EBH_6_FINAL_PRODUCTION_CODE_AUDIT_DOSSIER.md](EBH_6_FINAL_PRODUCTION_CODE_AUDIT_DOSSIER.md).

## Verification

```powershell
uv run pytest tests/qualification/ebh_6/ tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py -q
```

Log: `.tmp/session/ebh-6-p03-integrity/pytest.log`
