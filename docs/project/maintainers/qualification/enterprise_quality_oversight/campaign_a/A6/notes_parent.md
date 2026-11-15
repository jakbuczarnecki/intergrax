# A6 parent working notes (facts verified at BASELINE_SHA; used in W3 findings)
- 85 checks: 52 pytest_nodeid, 26 pytest_file, 7 python_script (ENT-A-BND-03, DOC-02, PRD-01, PRD-02, REG-03, SEC-01, SEC-02). None of the 85 entrypoints is under tests/qualification/ent_audit_x/acceptance.
- FRZ coverage labels (A6/coverage.jsonl): AUTOMATED 1, PARTIAL 85, NONE 139 (of 225). 85 PASS rows have no mapped check.
- env-gated wrappers: 86 test files under tests/ reference ENT_AUDIT_X_SEMANTIC (acceptance wrappers, `pytest.skip` unless env=1); 1 file references ENT_AUDIT_X_GATE_MATRIX; pyproject markers ent_semantic_gate/ent_semantic_matrix; run_qual_wave.py sets the env.
- existence-only gate: scripts/maintenance/check_architecture_debt_register.py checks file exists and substring "DEBT-" (L12-21).
- changed-since: impact_map.py _FAIL_CLOSED_PREFIXES covers only intergrax/platform/enterprise_audit/ and scripts/enterprise_audit/ (L15-18); selected_check_ids_from_impact (L335-341) unions prefix hits; unmapped product path selects nothing beyond the invariant floor (31 ids).
- dry-run results: FULL non-certifying dry-run: INCOMPLETE exit 2, 85 planned/0 executed; CHANGED-SINCE from planning pin: 68/85 planned, from 6bdf78c6: 80/85 (runner); build_impact_map: selected 68 and 86 ids?? (impact JSON counts include invariant floor ids), unmapped_paths 0 in both.
- exit codes: dry-run PARTIAL/INCOMPLETE -> exit 2 (EnterpriseAuditExitCode); FAST non-certifying run with only TRACKED_DEBT drift is promoted from PARTIAL to PASS/CLEAN (runner.py L267-274).
