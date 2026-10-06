# TRACE-X-P3-R1 — Governed boundary v2 certification

Qualification entrypoint: `tests/qualification/trace_x/test_trace_x_p3_r1_governed_boundary_v2.py`

Provenance baseline: `TRACE_X_P3_R1_START_HEAD = 72c1aca53663fe002faf2bb9bc4d5a14d2be92b1`

Corrective child baseline: `TRACE_X_P3_R1_R1_START_HEAD = 7bc5a7ec0fbacaf52558fdaf53384c20fa359e1b`

Scope: FRZ-TRC-04, FRZ-TRC-06 readiness (not independent closure).

R1-R1 hardening: canonical active TaskId + Governance tenant fail-closed on governed host;
GER/EBE v2 mandatory non-empty tenant; mechanical gates TXP3R1R1-Q01..Q17 in qualification module.

ADR: `docs/project/maintainers/architecture/ADR/ADR-TRACE-X-P3-R1-GOVERNED-BOUNDARY-V2.md`
