# PRE-ENT Wave 1 — ROADMAP-REPLAY-X mechanical gates (T-IGX-034-E)

| Field | Value |
| --- | --- |
| **QUALIFICATION_HEAD** | `f97d286b89b90d2f58b3c6cd288d629e3237bdaf` |
| **Ticket** | T-IGX-034-E |
| **Pytest** | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py` |

## Result

Mechanical exit gates reconciled @ HEAD post **T-IGX-034-B** (BND-03B manifest edge):

- **Historical evidence counter:** scoped prod-Δ SSOT `_pre_ent_l3_prod_delta.py` includes `applications/shipped_product_manifest_catalog.py`, `intergrax/applications/_shared/product_manifest_registry.py`, `intergrax/cli/doctor_health_app.py`, `intergrax/cli/doctor_diff_app.py` with disposition tied to PLAN T-IGX-034-B (no gate weakening).
- **VBN replay counter:** `_vbn_replay_counter.py` treats any path in `REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS` as expected prod delta since DUP-X; catalog path no longer triggers `unexpected_prod_delta_for_vbn`.
- **FRZ:** no PASS / promotion in this wave.

**Pytest @ `f97d286b`:** `16 passed` — `test_roadmap_replay_x_mechanical_exit_gates.py` (2026-11-10).
