# EC-W3-MAPPED-QUALIFICATION-05 corrective settlement

**PRE_G2_MAPPED_DIAGNOSTIC_EXECUTION — NOT FINAL QUALIFICATION**

- execution_sha: `f0cc4cf6e5c3a439762ccd7e17c96f2f2474c0df`
- starting_remote_sha: `f0cc4cf6e5c3a439762ccd7e17c96f2f2474c0df`

## Evidence integrity
- targeted 18/18 git blob matches: 18/18
- corrective 9/9 content matches: 9/9
- historical WB06: unresolved_historical_provenance preserved

## Mapped unit execution groups
- d01_architecture: exit=1 passed=50 failed=5 skipped=0
- d02_tenant_rag_context: exit=0 passed=38 failed=0 skipped=0
- d03_security: exit=1 passed=81 failed=2 skipped=0
- d04_evidence_reliability: exit=0 passed=48 failed=0 skipped=0
- d05_skills: exit=0 passed=65 failed=0 skipped=0

## Unique failed nodes
- tests.unit.applications._shared.test_fc1_harness_strict_persistence_wiring::test_acknowledgement_is_refused_when_hitl_tools_are_wired
- tests.unit.applications._shared.test_fc1_harness_strict_persistence_wiring::test_injected_human_decision_store_reaches_tool_wiring_context
- tests.unit.applications._shared.test_fc1_harness_strict_persistence_wiring::test_strict_harness_runtime_accepts_the_typed_acknowledgement
- tests.unit.applications._shared.test_fc1_harness_strict_persistence_wiring::test_strict_harness_runtime_refuses_missing_human_store_without_ack
- tests.unit.architecture.test_tier_dependency_boundaries::test_applications_do_not_import_other_applications
- tests.unit.runtime.security.test_p0_safety_7_sandbox_isolation_fail_closed::test_valid_sandbox_reaches_provider
- tests.unit.runtime.security.test_sec_planes_evol::test_security_bootstrap_discovers_security_defense_entry_point

## Obligations
- 34/34 obligations accounted in OBLIGATION_EXECUTION_MATRIX.json

## Remaining queue
- 29 qualification test files, 2 application host, 4 support modules, 12 documentary-only obligations

## ARCH-FREEZE
- locked uv sync with dev-ci extras completed in execution worktree

formal_certification_status: NOT_ESTABLISHED
