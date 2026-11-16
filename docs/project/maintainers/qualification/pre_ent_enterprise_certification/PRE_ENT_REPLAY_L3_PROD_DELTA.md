# PRE-ENT — REPLAY L3 production delta disposition (Wave 1)

| Field | Value |
| --- | --- |
| **QUALIFICATION_HEAD** | `01b3e1961f01e1ad578b84f3a20fbde7e16755f6` |
| **Prod baseline** | DUP-X acceptance `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |
| **Ticket** | T-IGX-034-E (mechanical gates; no FRZ PASS) |

## Batches

All prod paths under `intergrax/`, `agents/`, `applications/` changed since DUP-X @ QUALIFICATION_HEAD are **scoped** in `tests/qualification/roadmap_replay_x/_pre_ent_l3_prod_delta.py` + `_prod_delta_disposition.py`. Historical L2 CTRL-X evidence for `dependency_attempt_execution_boundary.py` remains @ qualification head `91649e5`.

| Batch | Disposition summary |
| --- | --- |
| T-IGX-033-A | Tier-0 CLI demo composition moved to Tier-3 (`intergrax/cli/*`, `applications/governed_contractor_application/*`) |
| T-IGX-030-B | Runtime error code contract + tool operation recovery |
| T-IGX-030-C | Nexus invoker `dep_boundary` `pool.submit` / `BaseException` handling |
| T-IGX-032-A | Skill execution pinning KV vs document store precedence |
| CTRL-X L2 | Pyright callback typing on `dependency_attempt_execution_boundary.py` (regression `tests/qualification/control_planes/ctrl_x/`) |
| Integration providers | VBN-neutral provider bundle/manifest SSOT under `intergrax/integrations/providers/` |
| Enterprise audit | `intergrax/platform/enterprise_audit/*` scaffolding on `development` |
| Adjunct Tier-0 | `health_score_wiring`, `sqlite_conversational_memory_store`, `error_codes` |
| T-IGX-034-B | BND-03B: `product_manifest_registry.py`, `applications/shipped_product_manifest_catalog.py`, `intergrax/cli/doctor_*_app.py` (catalog via `importlib` in handler scope) |

**Unscoped prod delta:** empty @ QUALIFICATION_HEAD → `historical_evidence_invalidated_by_later_changes` **0** (mechanical). VBN replay counter uses the same scoped path set (`_vbn_replay_counter.py` ⊆ `REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS`).
