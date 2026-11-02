# ANNEX C — Enterprise audit control matrix (complete)

**START_HEAD:** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**Total checks:** 85 (= 21×3 area + 13 TEN + 9 VBN)  
**Authority:** [ENT_AUDIT_X_COMPLETE_IMPLEMENTATION_AND_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_COMPLETE_IMPLEMENTATION_AND_CLOSURE_MASTER_PLAN.md)

Column key: **Gate** = existing nodeid/entrypoint; **Miss** = missing ENT adapter @ HEAD; **Batch** = implementation batch.

---

## Mandatory TEN checks (13/13)

| Check ID | Requirement | Semantic owner | Source paths | Gate | Miss | Technique | Positive | Negative | FRZ / SYS-INV | Evidence | Batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TEN-IDENTITY | Canonical tenant identity where required | TENANT-X | `tests/qualification/tenant_x/` | `test_tx_r1_head_q01_inventory_reconciles_twenty_two_mechanisms` | adapter | pytest RO | TENANT-X R1 GREEN @ SHA | `test_ten_identity_ent_skip_neg` | FRZ-TEN-01 | junit + manifest | B5 |
| TEN-PROPAGATION | Tenant propagated across boundaries | TENANT-X / TRACE-X | `tests/qualification/tenant_x/test_tenant_x_p0_current_head.py` | `test_tx_p0_head_q03_propagation_edges_reference_known_sources` | adapter | pytest RO | P0 propagation GREEN | `test_ten_propagation_drop_neg` | FRZ-TEN-02 | junit | B5 |
| TEN-AUTHORITY | Child scope ⊆ parent | GOV-X2 / TENANT-X | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py` | `test_tx_r4_q07_adversarial_background_identity` | adapter | adversarial pytest | R4 GREEN | widen scope inject | FRZ-TEN-03 | adversarial log | B5 |
| TEN-STATE | Cross-tenant state blocked | STATE-X / TENANT-X | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py` | `test_tx_r4_q05_adversarial_context_llm_cross_tenant` | adapter | adversarial | cross-tenant denied | allow cross-tenant state | FRZ-TEN-04 | finding BLOCKER | B5 |
| TEN-PROVIDER | Provider/profile preserve tenant | CONFIG-X / TENANT-X | `tests/qualification/existing_capability_configuration/` | adversarial tenant/provider tests | adapter | qual suite | INT-CONFIG cert paths replay | mismatched tenant pass | FRZ-TEN-05 | qual artifact | B5 |
| TEN-CONFIG | Tenant-scoped configuration | CONFIG-X | `tests/qualification/config_x/` | config_x reconciliation | adapter | pytest RO | CONFIG effective = tenant | global config fallback | FRZ-TEN-05 | config gate log | B5 |
| TEN-CREDENTIALS | No cross-tenant credential resolution | TENANT-X-R2 | `tests/qualification/tenant_x/test_tenant_x_r2_current_head.py` | `test_tx_r2_head_q01_qualification_closure_on_current_head` | adapter | pytest | R2 GREEN | credential bleed inject | FRZ-TEN-06 | junit | B5 |
| TEN-EVIDENCE | Evidence cannot establish other tenant truth | TRACE-X / TENANT-X | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py` | `test_tx_r4_q09_adversarial_functional_evidence_tenant` | adapter | adversarial | evidence scoped | accept wrong-tenant evidence | FRZ-TEN-07 | finding | B5 |
| TEN-TRACE | Trace/diagnostic tenant attribution | TRACE-X | `tests/qualification/trace_x/` | P3 tenant rows | adapter | qual subset | trace qual GREEN | mixed-tenant trace | FRZ-TEN-07 | trace junit | B5 |
| TEN-RECOVERY | Recovery/replay keeps tenant | STATE-X / TENANT-X | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py` | `test_tx_r4_q08_adversarial_vendor_async_queue` | adapter | adversarial | recovery continuity | wrong tenant resume | FRZ-TEN-08 | state+tenant log | B5 |
| TEN-ASYNC | Async resume tenant continuity | TENANT-X-R4 | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py` | `test_tx_r4_q06_adversarial_context_view_request_mismatch` | adapter | adversarial | async chain OK | context swap | FRZ-TEN-03 | R4 log | B5 |
| TEN-CROSS-TENANT | Legal cross-tenant needs explicit authority | GOV-X2 | `tests/qualification/gov_x2/` + TENANT-X R1 | legal cross-tenant = 0 | adapter | qual | GOV-X2 GREEN | implicit admin bypass | FRZ-TEN-09 | gov qual | B5 |
| TEN-FAIL-CLOSED | Missing/invalid tenant fails closed | TENANT-X | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py` | `test_tx_r4_q12_fail_closed_unknown_synthetic_context_caller` | adapter | adversarial | fail-closed | silent default tenant | FRZ-TEN-10 / SYS-INV tenant | R4 negative | B5 |

---

## Mandatory VBN checks (9/9)

| Check ID | Requirement | Semantic owner | Source paths | Gate | Miss | Technique | Positive | Negative | FRZ | Evidence | Batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VBN-INVENTORY-DRIFT | Inventory/matrix vs baseline | EBH-5 | `tests/qualification/ebh_5/` | `test_ebh5_g06_01_roadmap_31_domains_accounted` | adapter+hash | inventory digest diff | g06 PASS @ SHA | stale inventory PASS | FRZ-RPL-01 | drift JSON | B6 |
| VBN-CORE-HARDCODING | PROHIBITED unresolved = 0 | EBH-6 | `tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py` | `test_ebh6_prohibited_unresolved_zero` | adapter | pytest RO | register GREEN | inject prohibited | FRZ-RPL-02 | register export | B6 |
| VBN-SHADOW-SELECTION | No shadow provider selector | CONFIG-X | `tests/qualification/config_x/test_config_x_final_reconciliation_gates.py` | config final gates | adapter | pytest | CONFIG reconciled | shadow registry | FRZ-CFG-03 | gate log | B6 |
| VBN-SILENT-FALLBACK | No silent vendor fallback | Integrations/CONFIG | `tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py` | reconciliation tests | adapter | pytest | fail-closed proof | silent None fallback | FRZ-CFG-04 | ebh6 log | B6 |
| VBN-CONTRACT-LEAKAGE | No vendor concrete in contracts | EBH-2I | `tests/unit/architecture/test_ebh_2i_final_rescan_gate.py` | contract purity gate | adapter | AST+pytest | 2i GREEN | vendor type in contract | FRZ-CTR-04 | gate log | B6 |
| VBN-SQLITE-EFFECTIVE | SQLite effective authority qualified | EBH-5 annex | `tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py` | sqlite gates | adapter | pytest | annex PASS | unauthorized sqlite owner | FRZ-RPL-05 | ebh5 log | B6 |
| VBN-OBS-OPS-COUPLING | Obs/diag vendor replaceability | HARNESS-W5 | `tests/qualification/harness_01/` + W5 qual | OTLP replaceability | adapter | qual subset | W5 scoped PASS | hardcoded transport | FRZ-OBS-03 | W5 evidence | B6 |
| VBN-REPLACEABILITY-MATRIX | RPL-A..H integrity | EBH-5 | `tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py` | RPL-E gates | adapter | pytest | matrix PASS | broken RPL row | FRZ-RPL-06 | matrix JSON | B6 |
| VBN-REACHABILITY | Production-effective reachability | EBH-5 | `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py` | behavioral C | adapter | pytest | reachability proof | unreachable claimed ext | FRZ-RPL-07 | behavioral log | B6 |

---

## Area A–U (63 checks — 3 per area)

Each row: **Miss** = ENT `AuditCheck` adapter; **Technique** = RO pytest/script via sandbox; **Negative** = `tests/qualification/ent_audit_x/adversarial/test_<check_id>_neg.py`; **Batch** = B4 unless TEN/VBN module.

| Check ID | Requirement | Owner | Gate (representative) | FRZ families |
| --- | --- | --- | --- | --- |
| ENT-A-BND-01 | Illegal tier imports absent | EBH-3 / BND | `test_hardening_3_layer_boundary_gate` | FRZ-BND-* |
| ENT-A-BND-02 | Reverse dependency scan | EBH-3 | layer boundary gate | FRZ-BND-* |
| ENT-A-BND-03 | Agents/apps boundary | EBH-3 | tier import gates | FRZ-BND-* |
| ENT-A-OWN-01 | Exactly-one semantic owner | EBH-3 | `test_ebh_3_dependency_ownership_gate` | FRZ-OWN-* |
| ENT-A-OWN-02 | Composition owner sanctioned | EBH-4 | composition gates | FRZ-OWN-* |
| ENT-A-OWN-03 | Shadow authority detection | DUP-X | dup_x integrity | FRZ-OWN-* |
| ENT-A-CTR-01 | Public contract purity | EBH-2I | `test_ebh_2i_final_rescan_gate` | FRZ-CTR-* |
| ENT-A-CTR-02 | Contract/impl separation | EBH-2I | contract surface gate | FRZ-CTR-* |
| ENT-A-CTR-03 | Narrow responsibility | EBH-2I | rescan gate | FRZ-CTR-* |
| ENT-A-TYP-01 | Semantic Any at boundaries | EBH-2F | `test_ebh_2a_public_contract_boundary_gate` | FRZ-TYP-* |
| ENT-A-TYP-02 | type: ignore masking | EBH-2I | typing gates | FRZ-TYP-* |
| ENT-A-TYP-03 | String/dict pseudo-contracts | EBH-2I | dynamic dispatch gate | FRZ-TYP-* |
| ENT-A-DUP-01 | Duplicate factories | DUP-X | `test_dup_x_mechanical_exit_gates` | FRZ-DEBT / bypass |
| ENT-A-DUP-02 | Shadow registries | DUP-X | dup closed-world | REG |
| ENT-A-DUP-03 | Alternate execution paths | DUP-X | bypass gates | EXE/GOV |
| ENT-A-PLG-01 | Structural replaceability | EBH-5 | `test_ebh_5_factory_rpl_one_pass` | FRZ-PLG-* |
| ENT-A-PLG-02 | Provider behind contract | EBH-5 | RPL gates | FRZ-PLG-* |
| ENT-A-PLG-03 | No consumer patch replace | EBH-5 | behavioral proofs | FRZ-RPL-* |
| ENT-A-VBN-01 | Vendor inventory current | EBH-5/6 | g06 + hardcoding | FRZ-RPL-* |
| ENT-A-VBN-02 | Fallback classification | CONFIG-X | config_x gates | FRZ-CFG-* |
| ENT-A-VBN-03 | Reachability | EBH-5 | behavioral package | FRZ-RPL-* |
| ENT-A-GOV-01 | Governance ≠ Execution | GOV-X2 | `tests/qualification/gov_x2/` | FRZ-GOV-* |
| ENT-A-GOV-02 | Authority narrowing | GOV-X2 | gov qual | FRZ-GOV-* |
| ENT-A-GOV-03 | Fresh authorization | GOV-X2 | E2E cert tests | FRZ-GOV-* |
| ENT-A-EXE-01 | Single execution authority | Runtime EE | `test_ee_a1_execution_engine_ownership_certification_gate` | FRZ-EXE-* |
| ENT-A-EXE-02 | No peer schedulers | Harness | harness_01 gates | FRZ-EXE-* |
| ENT-A-EXE-03 | Canonical execution identity | TRACE-X | trace execution identity | FRZ-EXE-* |
| ENT-A-OBS-01 | Obs ≠ execution truth | HARNESS-W5 | harness W5 regressions | FRZ-OBS-* |
| ENT-A-OBS-02 | Event spine single | OBS ADR | observability gates | FRZ-OBS-* |
| ENT-A-OBS-03 | Correlation propagation | TRACE-X | trace qual | FRZ-TRC-* |
| ENT-A-STA-01 | Single semantic truth owner | STATE-X | state_x qual | FRZ-STA-* |
| ENT-A-STA-02 | Configured vs effective | STATE-X | state gates | FRZ-STA-* |
| ENT-A-STA-03 | Duplicate stores | STATE-X | R3 aux tests | FRZ-STA-* |
| ENT-A-SEC-01 | Secret boundary | SEC | `check_immutable_security_audit_trail.py` | FRZ-SEC-* |
| ENT-A-SEC-02 | Sensitive flow classification | SEC | security scripts | FRZ-SEC-* |
| ENT-A-SEC-03 | Redaction in reports | ENT runner | adversarial leak test | FRZ-SEC-* |
| ENT-A-REL-01 | Bounded admission | HARNESS-W4 | W4 qual slice | FRZ-REL-* |
| ENT-A-REL-02 | Retry caps | HARNESS-W2/W4 | harness gates | FRZ-REL-* |
| ENT-A-REL-03 | Overload classification | W4 | boundedness tests | FRZ-REL-* |
| ENT-A-REC-01 | Backup/restore responsibility | STATE-X | recovery qual | FRZ-REC-* |
| ENT-A-REC-02 | Restore no divergent truth | STATE-X | replay tests | FRZ-REC-* |
| ENT-A-REC-03 | Replay tenant continuity | TEN+STATE | tenant/state cross | FRZ-REC-* |
| ENT-A-PRD-01 | Health/readiness semantics | PROD-Q scoped | prod_q qual (Dimension A) | FRZ-PRD-* arch slice |
| ENT-A-PRD-02 | Degraded visibility | PROD-Q | scoped tests | FRZ-PRD-* |
| ENT-A-PRD-03 | Live slice limitation flag | ENT report | `QUALIFICATION_LIMITATIONS` | FRZ-PRD live OPEN |
| ENT-A-CMP-01 | No second permanent mechanism | COMPAT-X | compat_x qual | FRZ-CMP-* |
| ENT-A-CMP-02 | Deprecation enforcement | COMPAT-X | compat gates | FRZ-CMP-* |
| ENT-A-CMP-03 | Shim not authority | COMPAT-X | compat qual | FRZ-CMP-* |
| ENT-A-REG-01 | Stale allowlists = finding | QUAL-X/REPLAY | `test_replay_x_frz_row_count` | FRZ-REG-* |
| ENT-A-REG-02 | Deterministic qual | EVOL/QUAL | evol closed world | FRZ-REG-* |
| ENT-A-REG-03 | Env failure ≠ PASS | QUAL-X | qual infra tests | FRZ-REG-* |
| ENT-A-DOC-01 | Canonical docs present | REPLAY-X | `test_replay_x_mandatory_docs_present` | FRZ-DOC-* |
| ENT-A-DOC-02 | Ownership manifest drift | manifests | static diff | FRZ-DOC-* |
| ENT-A-DOC-03 | Roadmap/checklist sync | replay_x | roadmap drift zero test | FRZ-DOC-* |
| ENT-A-TEN-01 | TEN module rollup | TENANT-X | tenant_x final | FRZ-TEN-* |
| ENT-A-TEN-02 | Adversarial coverage | TENANT-X | R4 fast suite | FRZ-TEN-12 |
| ENT-A-TEN-03 | Continuous re-orchestration | ENT | TEN checks scheduled | FRZ-TEN-* |
| ENT-A-EVOL-01 | Impact-aware selection | EVOL-X | `test_evol_x_selection` | FRZ-EVOL-* |
| ENT-A-EVOL-02 | Conservative regression | EVOL-X | selection fault injection | FRZ-EVOL-06 |
| ENT-A-EVOL-03 | Post-freeze hooks | EVOL+guard matrix | dossier triggers test | FRZ-EVOL-09 |
| ENT-A-INV-01 | SYS-INV index coverage | Platform | `test_evol_x_closed_world_gates` | SYS-INV-* |
| ENT-A-INV-02 | Cross-family consistency | ECAP | coverage matrix audit | multi-FRZ |
| ENT-A-INV-03 | Fail-closed global | Platform | adversarial false PASS | SYS-INV fail-closed |

**Area row defaults (where not repeated):** Source paths = gate file directory; Miss = adapter; Technique = RO sandbox pytest/script; Positive = gate GREEN @ audit SHA; Negative = dedicated adversarial test; Evidence = per-check JSON in manifest; Acceptance = L2 PASS + FREEZE orchestration.

---

## Completeness

- Roadmap §3.1 categories → area + TEN + VBN rows above.  
- Unmapped mandatory TEN = 0; VBN = 0; areas A–U = 21×3.  
- **Total inventoried checks = 85.**
