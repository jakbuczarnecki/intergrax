# EC-W3-CORRECTIVE-06 corrective settlement

**PRE_G2_MAPPED_CORRECTIVE_EXECUTION — NOT FINAL QUALIFICATION**

- execution_sha: `272479cd2cff03ce28130115f5e5ba8286865185`
- starting_remote_sha: `bfecd8314c0990519a7e19733454d51c29a1e2e7`
- session: INTEGRAX | ENTERPRISE CAMPAIGN EQO | W3 B-1 | EC-W3-06

## Packages A–E
- Package A (environment PATH / uv discoverability): PASS
- Package B (cross-application policy isolation relocation): PASS
- Package C (sandbox authority positive control): PASS
- Package D (five-group mapped regression): executed per DES_PREREG
- Package E (evidence reconciliation): this run directory

## Mapped unit execution groups
- d01_architecture: exit=1 passed=52 failed=4 errors=0 skipped=0
- d02_tenant_rag_context: exit=0 passed=38 failed=0 errors=0 skipped=0
- d03_security: exit=0 passed=83 failed=0 errors=0 skipped=0
- d04_evidence_reliability: exit=0 passed=48 failed=0 errors=0 skipped=0
- d05_skills: exit=0 passed=65 failed=0 errors=0 skipped=0

## Corrected nodes (from EC-W3-05 red set)
- test_applications_do_not_import_other_applications: PASS (relocated witness)
- test_valid_sandbox_reaches_provider: PASS (explicit AUTH-C1a authority)
- test_security_bootstrap_discovers_security_defense_entry_point: PASS (uv on PATH)

## Residual — OPEN_HOST_POLICY_COMPOSITION_CORRECTIVE
- SkillBridgeCompositionError: policy fragment missing from host catalog: harness.policy_smoke
- Affects FC1 harness strict persistence wiring (4 tests) and MCP P4 lifecycle (1 test)
- No production or catalog mock changes in this batch

## Obligations
- 34/34 obligations accounted in OBLIGATION_EXECUTION_MATRIX.json

formal_certification_status: NOT_ESTABLISHED
terminal_state: CORRECTIVE_EXECUTION_PUBLISHED_WITH_EXACT_RESIDUALS
