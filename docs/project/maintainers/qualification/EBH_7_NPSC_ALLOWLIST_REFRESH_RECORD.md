# EBH-7 — NPSC allowlist refresh record (W3)

**Change class:** B — gate semantics aligned with DUP-X B4 (execution owner zone + materialize caller allowlist).

**Removed false positives:** substring _orchestration_backend matching host_orchestration_backend_spec_builder.

**Gate:** 	ests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py::test_npsc42_orchestration_backend_access_confined_to_allowlist

**Adversarial pack:** 	ests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py
