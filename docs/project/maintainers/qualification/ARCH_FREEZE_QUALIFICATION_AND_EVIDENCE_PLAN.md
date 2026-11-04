# ARCH-FREEZE qualification and evidence plan

**Execution SSOT:** [ARCH_FREEZE_COMPLETE_IMPLEMENTATION_AND_CERTIFICATION_MASTER_PLAN.md](./ARCH_FREEZE_COMPLETE_IMPLEMENTATION_AND_CERTIFICATION_MASTER_PLAN.md)  
**AUDIT_BASELINE_SHA:** `967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a`

| RUN_ID | TIER | TRIGGER | COMMAND_OR_WORKFLOW | ENVIRONMENT | TEST_SCOPE | EXPECTED_EVIDENCE | EXACT_SHA | REUSE_POLICY | FAILURE_POLICY | ACCEPTANCE_GATE | OWNER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1-GOV | L1 | product batch / dossier edit | `uv run python -m scripts.governance.igx_gov ticket` | Windows dev; repo venv | governance ticket + fast gate | ticket check log | IMPLEMENTATION_SHA | no | fail-closed stop | igx_gov exit 0 | implementer |
| L1-TRACE | L1 | dossier corrective batch | `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` | repo checkout | FRZ/WBS/gap/qual structural traceability | validator stdout | AUDIT_BASELINE_SHA or doc publication SHA | no | fail-closed stop | validator exit 0 | implementer |
| L1-LADDER | L1 | EVOL ladder delta @ baseline | manual: verify pins in `docs/project/maintainers/qualification/evol_x/EVOL_X_SHA_LADDER.md` | repo | link + SHA existence | L1 checklist note | `967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a` | reuse IC FAST only if no code delta | stop | pins resolve in git | implementer |
| L2-BND | L2 | post T-B2-BND03-RECONCILE | `uv run pytest tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py` | local venv | layer boundary gate | pytest junit / pass | IMPLEMENTATION_SHA | no | stop | tests green | tester |
| L2-DOC | L2 | post T-B3-DOC | `uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -k mandatory_docs` | local venv | FRZ-DOC mechanical docs | pytest pass | IMPLEMENTATION_SHA | partial historical | stop | tests green | tester |
| L2-ENT-ACCEPT | L2 | pre T-B4-L3-FULL | `uv run pytest tests/qualification/ent_audit_x/acceptance/ -q` | local venv | ENT acceptance wrappers | pytest pass | IMPLEMENTATION_SHA | no | stop | tests green | tester |
| L3-ENT-FULL | L3 | pre EVOL B6 (T-B4-L3-FULL) | `uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FULL --output-dir .tmp/proof/arch-freeze-full/ --evidence-bundle` | local venv | 85 manifest checks | `enterprise_audit_report.json` | TEST_EXECUTION_SHA | no | stop | checks_executed=85 PASS | tester |
| L3-ENT-FREEZE | L3 | post OD-ENT-01 (T-B7-L3-FREEZE) | `uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FREEZE --output-dir .tmp/proof/arch-freeze-freeze/ --evidence-bundle` | local venv; accepted baseline set | FREEZE vs baseline | FREEZE report + evidence_manifest | FREEZE_CERTIFICATION_SHA | no | stop | FREEZE PASS | independent_auditor |
| L3-NIGHTLY | L3 | optional CI consolidation | GitHub Actions workflow `.github/workflows/enterprise-audit-nightly.yml` @ HEAD | GitHub | enterprise nightly profile | CI run conclusion success | HEAD | historical only | investigate | workflow green | operator |
| L3-EVOL-REPLAY | L3 | EVOL B6 external gate | GitHub workflow dispatch / replay @ TES `3c4a641b3e48a13f437e830a0562e2a80b03aabb` (CI `41337077334`) | GitHub | EVOL qualification package | B6 package artifacts | pinned TES | yes if no code delta | stop | independent auditor sign-off | independent_auditor |
| L3-ENT-REPLAY | L3 | ENT B6 external gate | replay @ TES `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91`; B5 `300859537f12a1de7bf3d75fd685bc3a5c727183` | GitHub+local | ENT B6 dossier | B6 package | pinned TES | conditional | stop | independent auditor sign-off | independent_auditor |

**Note:** No L3 PASS is claimed by this planning document. Production reachability scan remains **NOT EXECUTED** — **T-B1-PROD-REACH-SCAN**.
