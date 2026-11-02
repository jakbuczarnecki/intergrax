# ANNEX G — Primary gate registry (ENT orchestration)

**START_HEAD:** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**Corpus size:** 411 files matching `test_*gate*.py` under `tests/` @ HEAD (mechanical count).

## Adapter qualification rules

| Rule | Description |
| --- | --- |
| D1 | Manifest row is SSOT; runner refuses unknown entrypoints (exit 3). |
| D2 | Duplicate invariant coverage: keep qualification SSOT (e.g. DUP-X over unit duplicate). |
| D3 | `mutating=false` default; gates needing writes must be `mutating=true` + isolated worktree only. |
| D4 | Missing env → exit **2** (qualification incomplete), never 0. |
| D5 | Parallel execution only within same `mutating` class and resource budget. |

## Primary orchestration entries (phase 1 manifest)

| ID | Entrypoint | Type | Mutating | Deps | Qual status @ HEAD | Evidence freshness |
| --- | --- | --- | --- | --- | --- | --- |
| PG-001 | `scripts/gates/check_application_production_gates.py` | script | RO | uv, dev-ci | EXISTS | replay @ audit SHA |
| PG-002 | `scripts/gates/check_audit_ideal_gates.py` | script | RO | uv | PARTIAL overlap | dedupe per D2 |
| PG-003 | `tests/qualification/ebh_5/test_ebh_5_g06_enterprise_integrity_gates.py` | pytest | RO | ebh_5 SSOT | QUALIFIED scoped | **replay required** |
| PG-004 | `tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero` | pytest | RO | ebh_6 | QUALIFIED scoped | replay required |
| PG-005 | `tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py` | pytest | RO | dup_x | CLOSED scoped | replay required |
| PG-006 | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py` | pytest | RO | replay_x | CLOSED scoped | replay required |
| PG-007 | `tests/qualification/config_x/test_config_x_final_reconciliation_gates.py` | pytest | RO | config_x | PASS historical | HEAD reconciliation |
| PG-008 | `tests/qualification/evol_x/gates/test_evol_x_selection.py` | pytest | RO | evol_x @ TES | TECH accepted | pin TES or replay |
| PG-009 | `tests/qualification/tenant_x/test_tenant_x_final_current_head.py` | pytest | RO | tenant_x | CLOSED @ 5ccb72fc | **replay @ audit SHA** |
| PG-010 | `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py` | pytest | RO | unit | EXISTS | replay |
| PG-011 | `tests/unit/architecture/test_ebh_2i_final_rescan_gate.py` | pytest | RO | unit | EXISTS | replay |
| PG-012 | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` | pytest | RO | unit | EXISTS | replay |
| PG-013 | `tests/qualification/ebh_7/` (allowlisted nodeids) | pytest-dir | RO | ebh_7 | CLOSED scoped | replay |
| PG-014 | `tests/qualification/gov_x2/` (allowlisted) | pytest-dir | RO | gov_x2 | CLOSED scoped | replay |
| PG-015 | `tests/qualification/state_x/` (allowlisted) | pytest-dir | RO | state_x | scoped PASS | replay |
| PG-016 | `tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py` | pytest | RO | ebh_5 | QUALIFIED | replay |
| PG-017 | `tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py` | pytest | RO | ebh_5 | QUALIFIED | replay |
| PG-018 | `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py` | pytest | RO | ebh_5 | QUALIFIED | replay |
| PG-019 | `tests/qualification/compat_x/` | pytest-dir | RO | compat_x | scoped | replay |
| PG-020 | `tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py` | pytest | RO | evol_x | TECH accepted | replay |

**Phase 2:** manifest generator ENT-T-0303 indexes remaining gate files into secondary tier (FAST excludes by policy).
