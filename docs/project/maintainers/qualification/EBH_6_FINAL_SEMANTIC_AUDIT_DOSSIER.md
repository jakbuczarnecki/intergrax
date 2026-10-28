# EBH-6 — Final Semantic Audit Dossier (mechanism-first)

| Field | Value |
| --- | --- |
| **Status** | **PARTIAL / EXACT BLOCKERS** — candidate handoff to **Protocol 03**; **not** EBH-6 CLOSED |
| **Audit mode** | ECAP §2.C mechanism review on closed-world L3 inventory (not cell-certification) |
| **EXPECTED_START_HEAD** | `d9f6b2a1b6ce600e4f1df8ecafc9c4d2790d5c45` |
| **IMPLEMENTATION_SHA** | `d051f5843d731d539a32eaaa9079f8f50aebace6` |
| **TEST_EXECUTION_SHA** | `d051f5843d731d539a32eaaa9079f8f50aebace6` |
| **CERTIFICATION_HANDOFF_SHA** | `9a731157bb833fbbab605191f0114aaa65c83292` |

**Authorities (read @ audit):** Roadmap §3.0.8 / §5 · [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) · [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) · [VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md) · `SYSTEM_INVARIANTS.md` · `INTERGRAX_ARCHITECTURE_PRINCIPLES.md` · [EBH_6_SCOPE_RECONCILIATION_DOSSIER.md](EBH_6_SCOPE_RECONCILIATION_DOSSIER.md) · [EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) · [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md).

**SSOT (unchanged — not rebuilt):** [`_l3_reachability_evidence.py`](../../../tests/qualification/ebh_6/_l3_reachability_evidence.py) (48 mechanisms) · [`_l3_aj_group_reconciliation.py`](../../../tests/qualification/ebh_6/_l3_aj_group_reconciliation.py) · [`_family_l9_evidence.py`](../../../tests/qualification/ebh_6/_family_l9_evidence.py).

**Implementer rule:** No independent Protocol 03 PASS · no OD-EBH6-08 · no agent AUDIT PASS.

---

## 1. Executive summary

This dossier records a **semantic review of production-effective mechanisms** indexed in the existing L3 annex. It does **not** promote obligations based on test-id lists alone; each row ties **code entrypoint → composition owner → backend selection / fail-closed behavior → applicable A–J classes → existing gate or explicit gap**.

| Metric | @ HEAD (mechanical SSOT) |
| --- | ---: |
| L3 mechanisms inventoried | **48** (≥3 per roadmap layer) |
| A–J obligation **cells** (layer × class rollup) | **480** |
| **PASS_CANDIDATE** | **3** |
| **PARTIAL_CANDIDATE** | **52** |
| **EVIDENCE_REQUIRED** | **342** |
| **FINDING** (tracked) | **2** |
| **N/A — WITH EVIDENCE** | **81** |
| Distinct **(layer, class)** groups still `EVIDENCE_REQUIRED` | **114** |
| **PROHIBITED** unresolved @ frozen scope | **0** |
| Architecture families — independent disposition | **23 × PENDING** |

**Verdict:** **`PARTIAL / EXACT BLOCKERS`** — sufficient for **independent replay @ CERTIFICATION_HANDOFF_SHA**, insufficient for EBH-6 closure or global FRZ PASS.

---

## 2. Method (ECAP-aligned)

| Step | Action taken |
| --- | --- |
| Inventory | Reuse `_l3_reachability_evidence` — no new catalog |
| Code path | Spot-read production modules on cited entrypoints (composition, integrations registry, execution, persistence) |
| Test semantics | Gates named in L3 records verified as **behavioral** vs **register-only** per ECAP-P1 |
| A–J | Group reconciliation from `_l3_aj_group_reconciliation` — **114** gap groups, not 342 micro-tests |
| Findings | Only register-backed **RATIFIED** rows treated as known violations; no new PROHIBITED without code change (OD-EBH6-01) |

**Out of scope:** DUP-X · EBH-7 (51 NYQ) · live PROD-Q · production code edits.

---

## 3. Mechanism domains (semantic audit)

### 3.1 Configuration and provider selection

| Mechanism | Production entry | Composition / config owner | Backend selection | Fallback | A–J touch | Review | Adequate existing test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **L3-HST-03** | `runtime_config_bridge.py` → `materialize_runtime_config` | Application `_shared` + `HostProfile` / env bridges | Profile-driven slugs via integration bridges; LLM via `resolve_llm_adapter` | Harness `strict_harness`; missing integration slug → factory fail-closed (see `resolve_slug`) | C, I, H | **PARTIAL** — CONFIG-X mapping proven; not all host-profile materialization paths replayed | `test_ebh6_config_x_head_reconciliation` |
| **L3-INT-01..03** | `integrations/providers/*/bundle.py` | `integrations/registry/factory.py` + `IntegrationProfile` | `resolve_slug` → catalog `get_entry` → category contract instance | **No silent default** — `IntegrationConfigurationError` if slug/profile/env unset | A, B, C, H, J | **PARTIAL** on integrations×A/H/J; per-layer class A still **EVIDENCE_REQUIRED** elsewhere | EBH-5 inventory + reachability gates |
| **L3-CTR-01** | `runtime_config_bridge` + contract modules | Contracts tier / app bridge | Typed profiles only | Boundary gate fail-closed on illegal imports | B, F, H | **EVIDENCE_REQUIRED** @ contracts for most classes — tier gate ≠ vendor-neutral census | Hardening-3 contracts import gate |

**Invariant confirmed:** sanctioned provider materialization goes through **IntegrationProfile + registry factory**; missing configuration **raises**, does not silently pick a vendor (`intergrax/integrations/registry/factory.py` `resolve_slug`).

**Invariant gap:** Roadmap §3.0.8 classes **A/B/D** on non-integration layers still lack **behavioral** selection-authority proofs beyond register exit counts.

---

### 3.2 Composition roots and adapter materialization

| Mechanism | Production entry | Owner | Effective backend | Fallback | A–J | Review | Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **L3-CMP-01** | `materialize_runtime_config` | TRACE-X sanctioned root | Full runtime profile stack | Lab harness branches isolated | H, B | **PARTIAL** (roots registry) | `test_ebh6_exit_sanctioned_composition_roots_non_empty` |
| **L3-CMP-02** | `orchestration_topology_submission.py` | Execution composition root | MSE policy factory | Fail-closed if factory missing | H, B | **PARTIAL** | Same + TRACE-X closed-world |
| **L3-CMP-03** | `runtime/transport/bootstrap.py` | Transport composition | Message-bus bundle (lab path) | Documented non-prod | H → **FINDING** | **RATIFIED** BND-COMPAT-R3; `production_effective=no` | Reachability + register |
| **L3-APP-01** | `production_delegated_subtask_child_execution_wiring.py` | Application `_shared` | Child execution port wiring | Contract-enforced | H, B | **EVIDENCE_REQUIRED** (7 classes @ app_services) | TRACE-X P4 LLM wrap propagation |

**Production reachability:** Primary production graph uses **registered composition sites** (TRACE-X registry parity tests). Lab-only transport bootstrap is **explicitly excluded** from production effectiveness in L3 record.

---

### 3.3 Execution and governance

| Mechanism | Entry | Owner | Backend / authority | Fallback | A–J | Review | Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **L3-EXE-01** | `host_task.py` → strategy router | ExecutionRuntime | In-process strategy routing | No alternate execution identity | **B → PASS** @ execution layer group | **PASS_CANDIDATE** group | EE-A1 certification |
| **L3-EXE-02** | `graph_executor.py` | Nexus composition | Graph lifecycle executor | Single authority gates | B, H | **PARTIAL** / layer gaps remain | Execution identity gate |
| **L3-EXE-03** | `inference_composition.py` | Execution factory | Governed inference executor | **No production caller** (TRACE-X Q07) | H, B | **N/A production** — inventoried only | TX P4 Q07 |
| **L3-GOV-01..03** | governance middleware / UAEP | Governance plane | Policy / verification contracts | Fail-closed on identity minting | H, F | **PARTIAL** @ governance×H | NPSC42 H1 freeze |
| **L3-SMC-01..03** | Nexus graph / orchestration / loop | Semantic core | Execution + ADE seam | Q2-D2 typed seam debt | **F → FINDING** on SMC-03 | **RATIFIED** Q2-D2 | Register + EE-A1 intersect |

**Confirmed:** Governance planes do not mint execution identity (NPSC42 H1 tests on cited paths).

**Known violation (tracked, not PROHIBITED):** `EBH6-DEBT-Q2-D2` — `nexus_loop.py` generic projection vs `AuthoritativeDecisionExposure` typing (FRZ-TYP-03 debt, RATIFIED).

---

### 3.4 Persistence, cache, and recovery

| Mechanism | Entry | Owner | Backend | Fallback | A–J | Review | Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **L3-STA-01..02** | SQLite bundle + profile materialization | Integrations edge | SQLite adapter via profile slug | No generic-core sqlite authority | E, B, H | **PARTIAL** @ state_persistence×E/H groups | SQLite annex + RPL-G |
| **L3-STA-03** | Qualification touchpoint registry | EBH-5 annex | References production paths | N/A | E | **Qual-only** inventory | `test_ebh6_sqlite_findings_cover_annex` |
| **L3-REC-01..03** | `dependency_attempt_execution_boundary.py`, retry budget | Recovery plane | Policy-bound retry / admission | Bounded retry; no recovery god-object | D, H, B | **EVIDENCE_REQUIRED** on most recovery-layer classes | NPSC5e + W3-C4 on samples |

**Confirmed @ sample paths:** recovery admission wired through **ResiliencePolicy** contracts; W3-C4 proves durable recovery canonical wiring on audited path.

**Gap:** Full **class D/E** behavioral matrices across all recovery-layer obligations remain **EVIDENCE_REQUIRED** (114-group theme).

---

### 3.5 RAG, memory, and tools

| Mechanism | Entry | Owner | Backend | Fallback | A–J | Review | Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **L3-RAG-01** | `rag_runtime_bridge.py` | Application + Integration materialization | Integration-owned vector/graph providers | Profile required | A, H, J | **EVIDENCE_REQUIRED** per mechanism controls (10/class rollup) | EBH-5 inventory |
| **L3-RAG-02** | `memory/user_profile_store.py` | Memory semantic owner | Contract ports | — | B, H | **EVIDENCE_REQUIRED** | Layer annex non-empty only |
| **L3-RAG-03** | `retrieval_service.py` | RAG service | Integration vectorstore | No RAG-local provider registry | A, C | **EVIDENCE_REQUIRED** | EBH-5 mechanical (scoped) |
| **L3-TLC-01..03** | tools registry / skill resolver | ToolRuntime SPI | Plugin registry wiring | Architecture gates | B, G, H | **EVIDENCE_REQUIRED** (9 groups @ tools layer) | DS / Hardening-5 plugin gates |

**Confirmed (historical scoped):** RAG retrieval does not embed provider choice in generic core (EBH-2G family evidence — **not** re-run as global PASS here).

---

### 3.6 Events, observability, and workers

| Mechanism | Entry | Owner | Spine | Fallback | A–J | Review | Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **L3-OBS-01..02** | `runtime_event_delivery_wiring.py`, `runtime_event_bus.py` | App + runtime observability | RuntimeEvent → bounded sink | Export failure isolated from execution | G, H | **PARTIAL** @ obs×G group | W5-B2 composition |
| **L3-OBS-03** | `diagnostic_orchestrator.py` | Diagnostics | Interpret-only ports | No diagnostic execution authority | G, H | **EVIDENCE_REQUIRED** @ most obs classes | CTRL-X R2 (sample) |
| **L3-TRC-01..03** | Event delivery + export sink + TRACE-X registry | Trace spine | Correlation fields preserved | — | H, G | **EVIDENCE_REQUIRED** (30 cells on TRC mechanisms alone) | TRACE-X P4 parity |
| **L3-BGI-01..03** | Worker / queue adapters (see L3 annex) | Background workers layer | Celery / queue bundles via integration factory | Profile-bound | A, H, J | **EVIDENCE_REQUIRED** | EBH-5 reachability (partial) |

**Confirmed @ W5 scope:** observability export does not mint execution truth (W5-B2 / W5-G profile tests on cited wiring).

---

## 4. Reconciling **342 EVIDENCE_REQUIRED** (grouped — not 342 tests)

### 4.1 Mechanical mapping

- **114** distinct `(layer, detection_class)` groups in `_ADJUDICATION` default path hold **`EVIDENCE_REQUIRED`**.
- Each applicable class on each L3 mechanism inherits the **group outcome** → **342** cell instances across **48** mechanisms (~7–10 EVIDENCE cells on heavily applicable layers such as `events_traceability`, `rag_memory`, `background_workers`).

### 4.2 Shared gap themes (one invariant → many cells)

| Theme ID | Invariant missing @ HEAD | Typical classes | Layers affected | Shared partial proof (does **not** close gap) |
| --- | --- | --- | --- | --- |
| **GAP-REG** | Register exit (unclassified=0, prohibited=0) ≠ class behavioral proof | A–J (all) | All 16 audited layers | `test_ebh6_prohibited_unresolved_zero` |
| **GAP-A-VENDOR** | Production-effective **vendor/authorship** of selection | **A** | 13 layers | EBH-5 inventory (**integrations** group PARTIAL only) |
| **GAP-B-HARD** | Layer-wide **hardcoding / bypass** census | **B** | 10 layers | EE-A1 (**execution×B** PASS only) |
| **GAP-H-REACH** | Independent **reachability replay** per layer | **H** | 9 layers | L3 annex index + sanctioned roots (PARTIAL groups) |
| **GAP-J-RPL** | **Replaceability behavioral** proof | **J** | 14 layers | RPL-G empty blockers (**integrations×J** PARTIAL) |
| **GAP-C/I-CFG** | Host/config materialization audit | **C**, **I** | 14 each | CONFIG-X 54/54 mapping (hosting PARTIAL) |
| **GAP-D-REC** | Recovery / durability semantics | **D** | 12 layers | NPSC5e/W3 on **samples** only |
| **GAP-F-TYP** | Typing / contract leakage | **F** | 13 layers | Register ratified debt rows (PARTIAL) |
| **GAP-G-OBS** | Observability authority boundaries | **G** | 9 layers | W5-B2 wiring (**obs×G** PARTIAL) |
| **GAP-E-STA** | Shadow store / persistence authority | **E** | 6 layers | SQLite annex (**state_persistence×E** PARTIAL) |

**Auditor note:** Closing a **theme** requires a **behavioral gate** whose proven invariant matches the theme — not re-labeling cells PASS.

### 4.3 Legitimate multi-cell proof (already adjudicated — not expanded)

| Bundle | Closes (group level) | Does **not** close |
| --- | --- | --- |
| EE-A1 P0 inventory | `execution×B` **PASS_CANDIDATE** | Other layers' class B |
| CONFIG-X head mapping | `runtime_hosting×C/I` **PARTIAL** | 14 other layers' C/I |
| EBH-5 inventory + RPL-G | `integrations×A/J` **PARTIAL** | Non-integration layers |
| NPSC42 H1 | `governance×H` **PARTIAL** | Full GOV-X2 replay |
| Register ratified debt | `*×F` **PARTIAL** on selected layers | FRZ-TYP behavioral re-audit |

**Remaining mandatory evidence groups:** **114** (`EBH6-P03-AJ-{layer}-{class}`).

---

## 5. Confirmed invariants (production paths sampled)

| ID | Statement | Mechanism evidence |
| --- | --- | --- |
| **INV-CFG-01** | Missing integration slug → **fail-closed** (`IntegrationConfigurationError`) | `integrations/registry/factory.py` |
| **INV-CMP-01** | Production composition sites are **registry-backed** (TRACE-X) | L3-CMP-01/02 + mechanical exit |
| **INV-EXE-01** | EE-A1 P0: **zero production bypass** on certified execution inventory | L3-EXE-01 + EE-A1 gate |
| **INV-GOV-01** | Governance does not mint execution identity | L3-GOV-* + NPSC42 H1 |
| **INV-OBS-01** | Observability spine non-authoritative for execution truth | L3-OBS-* + W5 gates |
| **INV-REG-01** | Frozen-scope **PROHIBITED unresolved = 0** | Findings register gates |

---

## 6. Real violations and tracked findings

| Finding ID | Location | Class | Production-effective | Status | Adequate test |
| --- | --- | --- | --- | --- | --- |
| **EBH6-DEBT-Q2-D2** | `intergrax/runtime/nexus/nexus_loop.py` (ADE projection) | **F** | yes | **RATIFIED** debt (FRZ-TYP-03) | Register + L3-SMC-03 FINDING row |
| **EBH6-DEBT-BND-COMPAT-R3** | `intergrax/runtime/transport/bootstrap.py` lab reachability | **H** | **no** (documented) | **RATIFIED** lab-only | L3-CMP-03 + register |

**PROHIBITED unresolved @ EBH-6 frozen scope:** **0**.

No new findings registered in this doc-only batch (no OD-EBH6-01).

---

## 7. Architecture families (23) and L9 blockers (8)

**Separation:** [`_family_l9_evidence.py`](../../../tests/qualification/ebh_6/_family_l9_evidence.py) holds **implementer proposals**; [`INDEPENDENT_AUDITOR_DISPOSITION`](../../../tests/qualification/ebh_6/_family_catalog.py) remains **23 × PENDING**.

| L9 blocker | Family | Semantic content of cited tests (code read) | Implementer proposal | Independent status |
| --- | --- | --- | --- | --- |
| **EBH6-BLK-L9-OWN-01** | OWN | EE final arch **owner matrix documented** vs facts module | SCOPED CANDIDATE | **PENDING** |
| **EBH6-BLK-L9-EXE-01** | EXE | EE-A1 **P0 bypass inventory = 0** on certified paths | PASS_CANDIDATE | **PENDING** |
| **EBH6-BLK-L9-GOV-01** | GOV | NPSC42 H1: **no policy engine in execution**; governance **no execution identity** | PASS_CANDIDATE | **PENDING** |
| **EBH6-BLK-L9-REC-01** | REC | NPSC5e **no recovery god-object** + W3-C4 **durable recovery wiring** | SCOPED CANDIDATE | **PENDING** |
| **EBH6-BLK-L9-SEC-01** | SEC | CTRL-X R1 **security composition** on audited control plane | SCOPED CANDIDATE | **PENDING** |
| **EBH6-BLK-L9-REL-01** | REL | EE-B1-2 **capacity architecture** gate | SCOPED CANDIDATE | **PENDING** |
| **EBH6-BLK-L9-HRN-01** | HRN | Harness/agent reference **boundary** qualification bundle | SCOPED CANDIDATE | **PENDING** |
| **EBH6-BLK-L9-TEN-01** | TEN | **Tenant scope** enforcement unit gate | SCOPED CANDIDATE | **PENDING** |

**FRZ alignment:** Families map to checklist rows per bundle `frz_criteria` — **no global FRZ PASS** claimed.

---

## 8. CONFIG-X and historical SHA

| Item | Value |
| --- | --- |
| Historical CONFIG-X PASS | Preserved @ `fac9a43f…` (FRZ-CFG — not revoked) |
| Current-HEAD mapping | **54/54** concerns mapped (`test_ebh6_config_x_all_concerns_mapped_at_head`) |
| Drift | Documented as **reconciliation**, not automatic VBN PASS |

---

## 9. Mandatory blockers remaining for EBH-6 closure

| Blocker | Count / ID | Owner |
| --- | --- | --- |
| Operator decisions | **OD-EBH6-01**, **OD-EBH6-08** | OPEN |
| Protocol 03 independent audit | **L9** + VBN candidate matrix | **PENDING** |
| Class-specific A–J evidence | **114 groups** / **342 cells** | `EBH6-P03-AJ-*` |
| L9 family blockers | **8 × EBH6-BLK-L9-*** | Auditor **PENDING** |
| Replaceability matrix | **51 NYQ** | EBH-7 (deferred) |
| Architecture families | **23 × PENDING** disposition | Independent auditor |

**Total implementer-tracked mandatory blockers @ HEAD:** **114 + 8 + 2 OD + 51 NYQ (program) + 23 family dispositions** — closure requires independent acceptance, not implementer tally reduction.

---

## 10. Candidate disposition for Protocol 03

Implementer submits this dossier + [EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md](EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md) + mechanical pytest log @ **CERTIFICATION_HANDOFF_SHA** for adversarial replay.

**Banner:** **`PARTIAL / EXACT BLOCKERS`** — **not** `READY FOR INDEPENDENT AUDIT` as closure claim; auditor may still accept **handoff readiness** for replay.

---

## 11. Verification (light)

```powershell
uv run pytest tests/qualification/ebh_6/ tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py -q
```

Log: `.tmp/session/ebh-6-semantic-audit/pytest.log`
