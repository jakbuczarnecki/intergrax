# EBH-6 — Current-HEAD Vendor Neutrality Audit (VBN §2.7 rollup)

| Field | Value |
| --- | --- |
| **Purpose** | Current-HEAD rollup per [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) §2.7 |
| **EXPECTED_START_HEAD (P03 integrity batch)** | `1858eafb84cfcc6c9197556bb24804e6bce864d8` |
| **IMPLEMENTATION_SHA** | `bccaf6ed24ca443ddf8d8ad6f4463e709b0f7289` |
| **TEST_EXECUTION_SHA** | `bccaf6ed24ca443ddf8d8ad6f4463e709b0f7289` |
| **CERTIFICATION_HANDOFF_SHA** | `d7b6442887941b6941a291b0efcf1271b9aee1de` |
| **Historical IMPLEMENTATION (pre-cert)** | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` — **HISTORICAL** |
| **PUBLICATION_SHA (P0–P6)** | `50d3b7785c7e13dbc6708143e7484a2eeea5eaca` — **HISTORICAL** |
| **CONFIG-X historical PASS** | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` — preserved |
| **EBH-5 independent audit** | `33ecf19cc020b112d9e586c87b49e819202f5d6d` — scoped PASS |
| **Disposition** | **CANDIDATE FOR INDEPENDENT PROTOCOL 03** — implementer does **not** set audit PASS |

## Mandatory audit layers (roadmap §3.0.8)

16 layers inventoried @ HEAD via [`tests/qualification/ebh_6/_closed_world_discovery.py`](../../../tests/qualification/ebh_6/_closed_world_discovery.py) and [`EBH_6_ENTERPRISE_SURFACE_INVENTORY.md`](EBH_6_ENTERPRISE_SURFACE_INVENTORY.md).

Register rows **`EBH6-LAYER-01..16`**: status **`L9_EVIDENCE_SUBMITTED`** (not independent RESOLVED).

L3 per-mechanism annex: [`_l3_reachability_evidence.py`](../../../tests/qualification/ebh_6/_l3_reachability_evidence.py) (≥3 mechanisms/layer).

Grouped A–J adjudication: [`_l3_aj_group_reconciliation.py`](../../../tests/qualification/ebh_6/_l3_aj_group_reconciliation.py) — **no unconditional PASS promotion (P03-01)**.

## A–J obligation cells (@ IMPLEMENTATION_SHA)

Mechanical census @ **bccaf6ed24ca443ddf8d8ad6f4463e709b0f7289** (57 pytest passed):

| Outcome | Cells (480 total) |
| --- | ---: |
| PASS_CANDIDATE | 3 |
| PARTIAL_CANDIDATE | 52 |
| EVIDENCE_REQUIRED | 342 |
| FINDING | 2 |
| N/A — WITH EVIDENCE | 81 |

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

Consolidated dossier: [EBH_6_SCOPE_RECONCILIATION_DOSSIER.md](EBH_6_SCOPE_RECONCILIATION_DOSSIER.md).

## Verification

```powershell
uv run pytest tests/qualification/ebh_6/ tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py -q
```

Log: `.tmp/session/ebh-6-p03-integrity/pytest.log`
