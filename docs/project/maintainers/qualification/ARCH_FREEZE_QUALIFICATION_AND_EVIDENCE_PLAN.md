# ARCH-FREEZE qualification and evidence plan

**AUDIT_BASELINE_SHA:** `967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a`

| RUN_ID | TIER | TRIGGER | COMMAND | ENV | EXPECTED_EVIDENCE | EXACT_SHA | REUSE | FAILURE | OWNER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
L1-GOV | L1 | product batch | uv run python -m scripts.governance.igx_gov ticket | Windows dev | ticket+fast gate log | IMPLEMENTATION_SHA | no | fail-closed | implementer |
L1-FRZ-COUNT | L1 | B0 | uv run python mechanical FRZ script | repo | 225/26 proof | 967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a | yes doc-only delta | stop | implementer |
L1-LADDER | L1 | 967ef2f delta | verify evol_x/EVOL_X_SHA_LADDER.md pins | repo | L1 PASS | 967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a | reuse IC qual if green | stop | implementer |
L2-BND | L2 | post B2 | uv run pytest tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py | local | junit | IMPLEMENTATION_SHA | no | stop | tester |
L2-DOC | L2 | B3 | uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -k mandatory_docs | local | pass | IMPLEMENTATION_SHA | partial | stop | tester |
L2-ENT-ACCEPT | L2 | B4 | uv run pytest tests/qualification/ent_audit_x/acceptance/ -q | local | acceptance wrappers | IMPLEMENTATION_SHA | no | stop | tester |
L3-ENT-FULL | L3 | pre EVOL B6 | uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FULL --output-dir .tmp/proof/arch-freeze-full/ --evidence-bundle | local | enterprise_audit_report.json checks_executed=85 | TEST_EXECUTION_SHA | no | stop | tester |
L3-ENT-FREEZE | L3 | post OD-ENT-01 | uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FREEZE --output-dir .tmp/proof/arch-freeze-freeze/ --evidence-bundle | local | FREEZE vs accepted baseline | FREEZE_CERTIFICATION_SHA | no | stop | independent_auditor |
L3-NIGHTLY | L3 | optional consolidation | CI enterprise-audit-nightly.yml @ HEAD | GitHub | nightly green | HEAD | historical only | investigate | operator |
L3-EVOL-REPLAY | L3 | EVOL B6 | replay CI 41337077334 / TES 3c4a641b… | GitHub | EVOL B6 package | pinned TES | yes if no code delta | stop | independent_auditor |
L3-ENT-REPLAY | L3 | ENT B6 | replay TES 89c76eb7… + B5 30085953… | GitHub+local | ENT B6 package | pinned TES | conditional | stop | independent_auditor |
