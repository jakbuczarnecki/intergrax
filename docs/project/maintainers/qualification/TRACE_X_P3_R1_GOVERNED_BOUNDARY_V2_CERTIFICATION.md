# TRACE-X-P3-R1 — Governed boundary v2 certification

Qualification entrypoint: `tests/qualification/trace_x/test_trace_x_p3_r1_governed_boundary_v2.py`

Provenance:

- `TRACE_X_P3_R1_START_HEAD = 72c1aca53663fe002faf2bb9bc4d5a14d2be92b1`
- `TRACE_X_P3_R1_R1_START_HEAD = 7bc5a7ec0fbacaf52558fdaf53384c20fa359e1b`
- `TRACE_X_P3_R1_R1_Q1_START_HEAD = 22a0725fe397ffaf0d82a955a6bec4f0cf644a1b`
- `TRACE_X_P3_R1_R1_Q2_START_HEAD = 8b448c5d99c02cc695b73fba2a5c1fa3830ac526`

Scope: FRZ-TRC-04, FRZ-TRC-06 readiness (recommendation only — not independent closure).

## P3-R1-R1-Q1 mechanical qualification

- **P3-R1-R1-Q-BLK-01:** RESOLVED / pending independent audit
- **P3-R1-R1-Q-BLK-02:** RESOLVED / pending independent audit (exact authorization→effect binding; Branch B propagation via optional GR-8 persistence at external-work composition)
- **Production delta (Q1):** `intergrax/` and `applications/governed_contractor_application/host/` unchanged vs `TRACE_X_P3_R1_R1_Q1_START_HEAD`
- **Gate SSOT:** `tests/qualification/trace_x/_trace_x_p3_r1_support.py` (`P3R1GateEvidence`, `P3_R1_R1_GATE_REGISTRY`)
- **Pass 1:** set `TRACE_X_P3_R1_R1_PASS1=1` and run qualification entry + `test_p3_r1_r1_governed_identity_tenant_gate.py` (session hook validates `PASS1_MECHANICAL_NODEIDS`)
- **Pass 2:** `test_governed_execution_result.py`, `test_host_attestation_and_receipt.py`, `test_canonical_serialization.py`, `test_fresh_side_effect_authorization.py`, `test_governed_proof.py`

### Mechanical gate → evidence (TXP3R1R1-Q03..Q30)

See `P3_R1_R1_GATE_REGISTRY` in `_trace_x_p3_r1_support.py` for authoritative `gate_id → pytest nodeid → FRZ` mapping.

### Tracked freeze debt (not P3-R1-R1 closure)

- **Nodeid:** `applications/governed_contractor_application/tests/host/test_gr6_wire_production_decision_governance.py::test_strict_host_composition_wires_agent_boundary_and_integration`
- **Exception:** `ToolDependencyAttemptBoundaryMaterializationError`
- **Classification:** TRACKED FREEZE DEBT (Reliability/Composition/PROD-Q)

ADR: `docs/project/maintainers/architecture/ADR/ADR-TRACE-X-P3-R1-GOVERNED-BOUNDARY-V2.md`

**Status:** TRACE-X-P3-R1-R1-Q2 = READY FOR INDEPENDENT CLOSURE REVIEW (Cursor recommendation only; FRZ-TRC-06 not closed here).
