## PLAN `T-IGX-034-A` — PRE-ENT Wave 0 presence reconciliation

Campaign: PRE-ENT predecessor certification closure. This plan is Wave 0 only — no product commits, no PASS claims.

### 1. Inputs

- Plan SSOT: `pre-ent_certification_closure_0d7be74d.plan.md`
- Branch: `development`
- Evidence HEAD: recorded in ticket Checkpoint at execution time

### 2. Presence checks (read-only)

| ID | Verification |
|----|----------------|
| 030-B | `intergrax/contracts/runtime_error_code.py`, contract import in `tool_operation_recovery.py`, gate file unchanged |
| 030-C | `invoker.py` `dep_boundary is None` branch: `try`/`except BaseException` around `pool.submit` with `ext_op.mark_failed()` |
| 032-A | `resolve_skill_execution_pinning_wiring`: `document_store if kv_store is None else None` |
| 033-A | `intergrax/cli/external_work.py` receipt-only; `test_bnd03_tier0_upward_edges_gate.py` exists |
| BND-03B | `intergrax/applications/_shared/product_manifest_registry.py` still uses `importlib.import_module` on Tier-3 manifest modules |
| DUR-01 | `validate_skill_execution_pinning_durability_for_composition` + store `is_durable` implementations |
| F-015 | `tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` |

### 3. Lifecycle reconciliation rule

If implementing commits are on HEAD and P8 integration is an ancestor, tickets still marked `plan_approved` should advance to `quality_gate` or `ready_for_audit` only after Wave 2 qualification — not in Wave 0 edits unless operator batches lifecycle hygiene.

### 4. Outputs

- `WAVE0_PRESENCE_RECONCILIATION.md` with columns: item \| on HEAD? \| ticket lifecycle \| wave1 action
- Ticket `T-IGX-034-A.md` checkpoint table

### 5. Rollback

Delete Wave 0 governance/docs files only; no product rollback.
