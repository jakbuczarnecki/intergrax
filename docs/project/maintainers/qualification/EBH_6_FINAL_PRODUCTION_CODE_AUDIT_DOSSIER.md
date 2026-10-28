# EBH-6 — Final Production Code Audit Dossier

| Field | Value |
| --- | --- |
| **Status** | **EBH-6 B2 COMPLETE — PARTIAL / EXACT BLOCKERS** — closed-world + L3 per layer; **Protocol 03 OPEN** |
| **B2 EXPECTED_START_HEAD** | `83b0a26186b0c11430c916842d7d813554286921` |
| **IMPLEMENTATION_SHA** | *(set @ B2 qual commit)* |
| **TEST_EXECUTION_SHA** | *(set @ B2 qual commit)* |
| **CERTIFICATION_HANDOFF_SHA** | `3f0a25439270b173c63989d2fd948c82a8cff6af` |

**Companion:** [EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md](EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md) (mechanism inventory) · [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md).

**Rules:** No production edits (OD-EBH6-01 OPEN). No Protocol 03 PASS · no OD-EBH6-08 · no EBH-6 CLOSED. **23 × PENDING** independent family disposition unchanged.

---

## 1. Batch summary

| Batch | Scope | Outcome |
| --- | --- | --- |
| **A** | IntegrationProfile, `registry/factory`, composition roots, hosting config, shadow/fallback | Code read + EBH-5 reachability gates; **5** A–J groups promoted to **PARTIAL** (not PASS) |
| **B2** | T-LAY-* — 16-layer closed-world census + 49 L3 mechanisms | **`test_ebh_6_b2_layer_closed_world_integrity.py` PASS**; layers remain **L9_EVIDENCE_SUBMITTED** |
| **C** | Reconcile 48 L3 / 16 layers / 114→**109** gap groups / 23 families / 8 L9 BLK | SSOT `_l3_aj_group_reconciliation.py` updated with code-backed `invariant_proves` only |

---

## 2. BATCH A — configuration and backend selection (code findings)

### 2.1 `intergrax/integrations/registry/factory.py`

| Property | Production behavior |
| --- | --- |
| **Semantic owner** | Integrations / sanctioned composition |
| **Canonical contract** | `PlatformIntegrationContract` per category |
| **Resolution order** | explicit `slug` → `IntegrationProfile` → `INTERGRAX_INTEGRATION_*` env → **`IntegrationConfigurationError`** |
| **Materialization** | Catalog `get_entry(slug).factory(...)` only; `_require_platform_integration_contract` + category match |
| **Shadow registry** | No second global selector in factory module |
| **default_lab_profile** | Defined in contracts; **not** invoked from production `runtime_config_bridge` path (lab/tests only) |

**Tests:** `test_ebh5_reach_integration_unknown_slug_fail_closed`, `test_ebh5_reach_factory_module_is_sanctioned_composition_owner`, `test_ebh5_reach_env_slug_is_canonical_merge_not_shadow_registry`.

**A–J groups reconciled:** `integrations×B`, `integrations×C`, `integrations×I`, `composition_roots×C` → **PARTIAL_CANDIDATE** (factory slice only).

### 2.2 Application integration bridge

`intergrax/applications/_shared/integration_runtime_bridge.py` — attaches `env.integration_profile` / composition overlay to `RuntimeConfig.integration_profile` **without** alternate slug resolution.

**Limitation:** Does not close `runtime_hosting×I` or other layers' class **I** cells.

### 2.3 Runtime hosting / CONFIG-X

Prior **PARTIAL** on `runtime_hosting×C/I` retained (mapping inventory). Code audit did not find silent host-level backend defaults in `context_bootstrap` beyond profile-driven composition (full materialization replay still **EVIDENCE_REQUIRED** on other layers).

---

## 3. BATCH B — architecture boundaries (sampled)

| Area | Production check | Violations in scope |
| --- | --- | --- |
| **Execution / governance** | `host_task` router, NPSC42 governance middleware, graph executor | **None new**; Q2-D2 seam remains **RATIFIED** debt |
| **Persistence / recovery** | SQLite via integration bundles; `dependency_attempt_execution_boundary` | **None new**; annex accountability unchanged |
| **Contracts / typing** | `nexus_loop.py` lines ~703, ~1074 — `object \| None` projection | **EBH6-DEBT-Q2-D2** (FINDING on L3-SMC-03) |
| **RAG / memory** | RAG materialization via integration profile; EBH-5 RAG graph gate **skipped** @ HEAD | **`rag_memory×A` remains EVIDENCE_REQUIRED** |
| **Tools / agents** | Tier boundary + lifecycle gates (existing qual) | **EVIDENCE_REQUIRED** on most tool/agent class cells |
| **Events / OBS / workers** | W5 spine on app wiring; export isolated from execution | **EVIDENCE_REQUIRED** on trace/worker layers |

**Lab-only path:** `runtime/transport/bootstrap.py` — **EBH6-DEBT-BND-COMPAT-R3** (production_effective=no).

---

## 4. BATCH C — evidence reconciliation (@ HEAD)

### 4.1 A–J cell census (mechanical)

| Outcome | Cells |
| --- | ---: |
| PASS_CANDIDATE | 3 |
| PARTIAL_CANDIDATE | **67** (+15 vs pre-batch from group promotions) |
| EVIDENCE_REQUIRED | **327** (−15) |
| FINDING | 2 |
| N/A — WITH EVIDENCE | 81 |

### 4.2 Distinct gap groups

| Metric | Value |
| --- | ---: |
| **`EVIDENCE_REQUIRED` groups** | **109** (was 114) |
| **Code-audit promotions** | 5 groups → PARTIAL |

Promoted groups (code + existing test, **not** PASS):

1. `integrations×B` — factory contract enforcement  
2. `integrations×C` — fail-closed resolution order  
3. `integrations×I` — profile/env canonical merge  
4. `composition_roots×C` — sanctioned factory owner  
5. `reliability_recovery×H` — NPSC5e + L3 recovery sample  

**Not promoted:** `rag_memory×A` — EBH-5 RAG graph gate **skipped** @ HEAD (`integration_graph_store` absent).

**Remaining 109 groups:** mandatory `EBH6-P03-AJ-{layer}-{class}` — **not auto-closed**.

### 4.3 16 layers / 49 L3 records

B2 adds **`L3-RAG-04`** (`graph_store_bootstrap.py`) and corrects hosting L3 entrypoints to `runtime_config_bridge` / `environment_wiring` @ HEAD. Outcomes derive from group table + mechanism FINDING rows (Q2-D2, BND-COMPAT-R3).

### 4.4 23 families

Implementer bundles unchanged. **Independent disposition: 23 × PENDING.**

### 4.5 8 × L9 blockers (implementer gate semantics — auditor PENDING)

| Blocker | Code-audit alignment |
| --- | --- |
| EBH6-BLK-L9-EXE-01 | EE-A1 P0 inventory consistent with execution L3 samples |
| EBH6-BLK-L9-GOV-01 | Governance modules reviewed; match NPSC42 H1 tests |
| EBH6-BLK-L9-REC-01 | Recovery boundary modules match NPSC5e/W3 samples |
| EBH6-BLK-L9-OWN/SEC/REL/HRN/TEN-01 | No contradiction found in audited paths; **independent replay still required** |

---

## 5. Production violations

| ID | Location | Invariant | Disposition |
| --- | --- | --- | --- |
| EBH6-DEBT-Q2-D2 | `intergrax/runtime/nexus/nexus_loop.py` | FRZ-TYP-03 generic seam | **FINDING** / RATIFIED — **OD-EBH6-01** if remediated |
| EBH6-DEBT-BND-COMPAT-R3 | `intergrax/runtime/transport/bootstrap.py` | Unauthorized prod reachability | **FINDING** / RATIFIED lab-only |

**New PROHIBITED in frozen scope:** **none detected** in this audit batch.

---

## 6. Mandatory blockers (unchanged authority)

| Blocker | Status |
| --- | --- |
| OD-EBH6-01 | OPEN |
| OD-EBH6-08 | OPEN |
| Protocol 03 | OPEN |
| **109 × EBH6-P03-AJ-*** | OPEN |
| **8 × EBH6-BLK-L9-*** | Implementer submitted — auditor **PENDING** |
| **51 NYQ** | EBH-7 deferred |
| **23 family dispositions** | **PENDING** |

---

## 7. Verification

```powershell
uv run pytest tests/qualification/ebh_6/test_ebh_6_scope_reconciliation_integrity.py tests/qualification/ebh_6/test_ebh_6_l3_and_family_evidence_integrity.py tests/qualification/ebh_6/test_ebh_6_b2_layer_closed_world_integrity.py tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py -q
```

Log: `.tmp/session/ebh-6-b2/pytest.log`

---

## 8. Terminal banner

**`EBH-6 B2 — PARTIAL / EXACT BLOCKERS`**

Closed-world inventory + L3 verified for **16** layers; **109** mandatory A–J evidence groups remain for **B3**. **STOP before B3** per closure plan.
