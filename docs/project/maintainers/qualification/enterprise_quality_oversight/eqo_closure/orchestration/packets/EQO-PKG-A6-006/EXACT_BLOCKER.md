# EQO-PKG-A6-006 — exact blocker (EQOA-A6-006)

**Finding:** `EQOA-A6-006` (EVIDENCE_DEFICIENCY, E3)  
**Evidence base:** `8e146e21cd8c63a47d621edc05d5f0036ba0c9ee`  
**Representative test:** `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_bnd_01.py::test_acceptance_ent_a_bnd_01_semantic_gate`  
**Status:** `BLOCKED_EXACT_REASON`

## Disposition

The `pytest.skip` when `ENT_AUDIT_X_SEMANTIC` is unset remains **by design** per the ENT-AUDIT-X qualification contract. EQO does **not** retune this gate (no edit to the acceptance wrapper, `pyproject.toml` markers, or CI slice meaning).

| Field | Value |
| --- | --- |
| Owner | **ENT-AUDIT-X** / **PT-E03** (`T-ENF-3`, non-product) |
| Remediation lane | Acceptance conftest hook, certifying semantic automation, debt-gate rewrite (see WP12 `PT-E03`; closes A6-001 and A6-006 together) |
| EQO action | Record exact blocker only; no product-path change on this packet |

## Contract citation

| Source | Rule |
| --- | --- |
| `docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_ONE_PASS_AGENT_HANDOFF_B6.md` — **Wave 2 integration (security + R1/R2/R3 harness)** | Semantic acceptance (C033–C036): 85 wrappers call `_gate_oracle` (**skip unless `ENT_AUDIT_X_SEMANTIC=1`**). |
| Same file — **Qual evidence @ HEAD (pre-TES)** | Default slice: L1/L2/L3 pytest (**108 passed; semantic 85 matrix opt-in**). Semantic 85/85 runs only after `$env:ENT_AUDIT_X_SEMANTIC='1'` and `tests/qualification/ent_audit_x/acceptance`. |
| `docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md` — **Required checks (target)** | PR `ent-audit-fast` = L1/L2/L3 + CLI FAST; semantic acceptance is not in the default PR qualification slice. |
| `pyproject.toml` — pytest marker `ent_semantic_gate` | Documented opt-in: ENT-AUDIT-X semantic acceptance (85 CheckID gates; **`ENT_AUDIT_X_SEMANTIC=1`**). |

## Mechanism note (M1)

Campaign A mechanism M1 verdict stands: **SILENT_SKIP_BY_DESIGN** for env-gated semantic wrappers; they are qualification oracles for the certifying semantic wave, not manifest entrypoints executed in the default L1/L2/L3 slice.

**Ref:** `governance/tickets/T-IGX-046-F.md`