# EBH-6 — Scope Reconciliation Dossier (SCOPE-01..03)

| Field | Value |
| --- | --- |
| **Status** | **PARTIAL / EXACT BLOCKERS (OD-EBH6-Q2D2 OPEN; T2-AUD)** — tier **T1=2** @ **CERTIFICATION_HANDOFF_SHA**; **not** Protocol 03 PASS |
| **Closure plan authority** | [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) @ **PLAN_PUBLICATION_SHA** `0489054c5f80ee392211fb33a39f2736fa2fdfa0` |
| **Independent auditor disposition** | **23 × PENDING** — implementer must not set PASS |
| **P03 EXPECTED_START_HEAD** | `cc76a4df1f401c8d7c374ec3b8d2bd7f21461b97` |
| **IMPLEMENTATION_SHA** | **CERTIFICATION_HANDOFF_SHA** (mechanical: `_evidence_manifest.evidence_git_head_sha()`) |
| **TEST_EXECUTION_SHA** | **CERTIFICATION_HANDOFF_SHA** — `.tmp/session/ebh-6-t2-repair/pytest.log` |
| **CERTIFICATION_HANDOFF_SHA** | @ B5 cert handoff commit on `development` |

Mechanical SSOT: [`tests/qualification/ebh_6/_l3_aj_group_reconciliation.py`](../../../tests/qualification/ebh_6/_l3_aj_group_reconciliation.py), [`_layer_evidence.py`](../../../tests/qualification/ebh_6/_layer_evidence.py), [`_l3_reachability_evidence.py`](../../../tests/qualification/ebh_6/_l3_reachability_evidence.py), [`_family_l9_evidence.py`](../../../tests/qualification/ebh_6/_family_l9_evidence.py).

**Normative crosswalk (T-NORM-01 @ B1):** L9-A matrix in [EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md](EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md) ↔ `_layer_evidence.layer_aj_applicability` — **133** applicable pairs; B5 class gates + **T2 bundle review** ([EBH_6_T2_EVIDENCE_INDEPENDENT_REVIEW.md](EBH_6_T2_EVIDENCE_INDEPENDENT_REVIEW.md), `_t2_evidence_bundles.py`) + tier census (`_tier_audit.py`) — **`EVIDENCE_REQUIRED` = T2-AUD INSUFFICIENT** (not false scanner T2).

## Evidence tier audit @ HEAD

**Mechanical SSOT:** [`tests/qualification/ebh_6/_tier_audit.py`](../../../tests/qualification/ebh_6/_tier_audit.py) · optional session copy: `.tmp/session/ebh-6-cert-handoff/tier-audit.md`

| Tier | Groups @ B5 handoff | Outcome mapping |
| --- | ---: | --- |
| **T1** | **2** | `EVIDENCE_REQUIRED` (`contracts`×F, `application_services`×F) |
| **T2** | **129** | `PARTIAL_CANDIDATE` (T2-AUD **ACCEPTED** bundles) |
| **T3** | **2** | `PASS_CANDIDATE` (`execution`×B) + `FINDING` (`semantic_core`×F / OD-EBH6-Q2D2 OPEN) |

**133** applicable pairs · **49** L3 mechanism records · L3→A–J crosswalk: `l3_mechanism_to_aj_groups()` in `_tier_audit.py`.

**Mechanism-first semantic audit:** [EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md](EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md) (**PARTIAL / EXACT BLOCKERS** @ pre-B1 handoff SHA until B5).

Roadmap authority: [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) § EBH-6 / §3.0.8. VBN: [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md).

## Batch sequencing (corrected plan B2–B7)

| Batch | Scope | Exit (summary) |
| --- | --- | --- |
| **B1** | T-NORM-01, T-SHA-01, T-DOC-01 | L9-A ↔ `_layer_evidence`; single SHA ladder — **done** |
| **B2** | T-LAY-* (8 packages, 16 layers) | Verified closed-world inventory + L3 **per layer** — **done** (`test_ebh_6_b2_layer_closed_world_integrity.py`); **not** auto-RESOLVED |
| **B3** | T-CLS-* | **133** groups adjudicated; **2** `EVIDENCE_REQUIRED` (T2-AUD); class gates + reuse @ HEAD — **not** Protocol 03 PASS |
| **B4–B7** | Families, publication, P03, OD-EBH6-08 | See closure plan §6 — **no B6** before B5 zero gap groups |

## SCOPE-01 — A–J grouped reconciliation (@ IMPLEMENTATION_SHA)

| Outcome (133 groups) | Count |
| --- | ---: |
| **EVIDENCE_REQUIRED** (mandatory gap @ HEAD) | **2** (T1 tier — `contracts_typing` bundle) |
| **PARTIAL_CANDIDATE** | **129** (T2 tier) |
| **PASS_CANDIDATE** | 1 (`execution` × B — EE-A1) |
| **FINDING** | 1 (`semantic_core` × F — Q2-D2) |

L3 obligation cells (480 mechanism rollup): see [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md) — **not** the Protocol 03 counting unit.

**Positive proofs (group bundles):** register `unclassified=0`, `prohibited unresolved=0`; CONFIG-X 54/54 mapping; tier boundary gate; sanctioned roots; layer gate tests from `_layer_evidence.layer_gate_test_ids`.

**Adversarial negatives (unchanged):** `EBH6-DEBT-Q2-D2` (class F FINDING); `EBH6-DEBT-BND-COMPAT-R3` (class H FINDING, lab-only).

## SCOPE-02 — Families CMP / PRD / REG / DOC

| family | disposition | closure |
| --- | --- | --- |
| **CMP** | SCOPED CANDIDATE | COMPAT-X historical closure; EVOL consumer impact OUT OF SCOPE (Path B) |
| **PRD** | SCOPED CANDIDATE | Live PROD-Q vendor slice OUT OF SCOPE; architecture census via register |
| **REG** | SCOPED CANDIDATE | EBH-6 qual regression bundle — **not** QUAL-X / global FRZ-REG |
| **DOC** | SCOPED CANDIDATE | Mandatory docs + this dossier content gate — **not** global FRZ-DOC PASS |

## 16 layers — register status

**14 × CANDIDATE-RESOLVED**, **2 × L9_EVIDENCE_SUBMITTED** (`contracts`, `application_services` — class F T2-AUD gap). ≥3 L3 mechanisms each (49 total annex rows) — **not** independent RESOLVED (B6 acceptance only).

## B2 — T-LAY package outcomes (@ HEAD)

| TASK-ID | Layers | Inventory SSOT | L3 gate |
| --- | --- | --- | --- |
| **T-LAY-CTR-SMC** | contracts, semantic_core | `discover_contracts_surface_paths`, `discover_semantic_core_paths` | hardening-3, EE-A1, L3-CTR/SMC-* |
| **T-LAY-APP-CMP** | application_services, composition_roots | TRACE-X roots, wiring, composition hints | trace_x, L3-APP/CMP-* |
| **T-LAY-GOV-EXE** | governance, execution | `discover_governance_execution_paths` | npsc42, EE-A1, L3-GOV/EXE-* |
| **T-LAY-INT-STA** | integrations, state_persistence | bundles + sqlite annex | ebh5 reachability, L3-INT/STA-* |
| **T-LAY-REC-OBS** | reliability_recovery, observability_diagnostics | `discover_resilience_observability_paths` | npsc5e, w5-b2, L3-REC/OBS-* |
| **T-LAY-TRC-WRK** | events_traceability, background_workers | event spine + worker census | trace_x, ebh5, L3-TRC/BGI-* |
| **T-LAY-AGD-RAG** | agent_distribution, rag_memory | agents + `graph_store_bootstrap.py` | lifecycle gate, L3-AGD/RAG-* |
| **T-LAY-HST-TLC** | runtime_hosting, tools_skills_capabilities | hosting contracts bridge + tools/skills census | config_x, plugin gates, L3-HST/TLC-* |

## 23 families — implementer proposals

| disposition | count |
| --- | ---: |
| PASS_CANDIDATE | 2 (EXE, GOV) |
| SCOPED CANDIDATE | 21 |
| EVIDENCE REQUIRED | 0 |

## CONFIG-X reconciliation

@ HEAD: **54/54** concerns mapped — gate `test_ebh6_config_x_all_concerns_mapped_at_head`.

## Unresolved mandatory blockers (certification)

| ID | Status |
| --- | --- |
| **109 × EVIDENCE_REQUIRED groups** | **Superseded by T2-AUD** — honest **2×** mandatory @ HEAD (`contracts_typing`); see [EBH_6_T2_EVIDENCE_INDEPENDENT_REVIEW.md](EBH_6_T2_EVIDENCE_INDEPENDENT_REVIEW.md) |
| **16 × layer rows** | **B2 inventory+L3 verified** — `L9_EVIDENCE_SUBMITTED`; RESOLVED only B6 |
| **OD-EBH6-01** | OPEN (no production wave) |
| **OD-EBH6-08** | OPEN (parent closure) |
| **L9 Protocol 03** | OPEN — **not** B1 scope |
| **51 EBH-5 NYQ** | Deferred EBH-7 |
| **DUP-X** | Not started |
| **8 × EBH6-BLK-L9-*** | Independent PENDING |

## Explicit non-claims

- No **Protocol 03 PASS**, no **EBH-6 CLOSED**, no global **FRZ PASS**, no agent **AUDIT PASS**.
- B1 complete ≠ independent audit ready (B6).

## Verification (@ TEST_EXECUTION_SHA)

```powershell
uv run pytest tests/qualification/ebh_6/test_ebh_6_scope_reconciliation_integrity.py tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py tests/qualification/ebh_6/test_ebh_6_l3_and_family_evidence_integrity.py -q
```

Log: `.tmp/session/ebh-6-b1/pytest.log`
