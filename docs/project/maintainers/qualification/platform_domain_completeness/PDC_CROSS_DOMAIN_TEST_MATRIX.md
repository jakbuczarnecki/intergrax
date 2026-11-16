# PDC-X — cross-domain test matrix (PDC-P4-03)

**Generated** by `scripts/qualification/pdc_x_wbs.py`. Owner batch: **B-P6** (T-IGX-036-S). Every scenario is a real-integration qualification test: mandatory paths use the canonical composition roots, not mocks; fakes are allowed only for external vendors (Path B).

| XD | Pair | Scenario | Planned module | Requirements proven |
| --- | --- | --- | --- | --- |
| XD-01 | Agent ↔ Model | Activated agent revision pins profile/model; provider swap by config leaves the agent contract unchanged; capability mismatch fails closed | `tests/qualification/pdc_x/xd/test_xd_01_agent_model.py` | scenario-level (no single row) |
| XD-02 | Agent ↔ Context | Tenant A agent cannot read tenant B memory/RAG; revoked agent loses context access | `tests/qualification/pdc_x/xd/test_xd_02_agent_context.py` | PDC-D6-R019 |
| XD-03 | Agent ↔ Tools/Governance | Agent-proposed tool call without a governance permit is denied; no agent-owned execution path | `tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py` | scenario-level (no single row) |
| XD-04 | Agent ↔ Host | Configured vs effective roster; restart preserves active revision and profile digest | `tests/qualification/pdc_x/xd/test_xd_04_agent_host.py` | PDC-D1-R016 |
| XD-05 | Host ↔ Background | Shutdown drains in-flight tasks; scheduler lease released/fenced; no duplicate on restart | `tests/qualification/pdc_x/xd/test_xd_05_host_background.py` | PDC-D4-R005, PDC-D7-R014 |
| XD-06 | Background ↔ Security | Retried task re-authorizes; credential revoked mid-retry fails closed; no duplicate side effect | `tests/qualification/pdc_x/xd/test_xd_06_background_security.py` | PDC-D6-R009, PDC-D6-R019 |
| XD-07 | Model ↔ Context | Budget/truncation keeps mandatory instructions; streamed tool-call provenance recorded; restart reconstruction | `tests/qualification/pdc_x/xd/test_xd_07_model_context.py` | PDC-D2-R013 |
| XD-08 | Evaluation ↔ Model/Agent | Baseline vs candidate rerun is digest-identical; an evaluation PASS cannot authorize promotion | `tests/qualification/pdc_x/xd/test_xd_08_evaluation.py` | PDC-D5-R009, PDC-D5-R014 |
| XD-09 | Artifacts ↔ Context/Tools | Large tool output spills to an artifact ref; compaction keeps the ref; evidence artifacts undeletable | `tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py` | PDC-D2-R018, PDC-D8-R012, PDC-D8-R013, PDC-D8-R019 |
| XD-10 | Artifacts ↔ Security | Cross-tenant artifact read denied; TTL deletion honored; no raw secret in artifacts | `tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py` | PDC-D6-R019, PDC-D6-R022, PDC-D8-R005 |
| XD-11 | All ↔ Observability/Trace | Every mandatory operation emits canonical trace/evidence (inventory-driven assertion) | `tests/qualification/pdc_x/xd/test_xd_11_observability.py` | scenario-level (no single row) |
| XD-12 | All ↔ EVOL/ENT-AUDIT | ENT-AUDIT changed_since maps every PDC-touched path to checks (impact-map completeness) | `tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py` | scenario-level (no single row) |
