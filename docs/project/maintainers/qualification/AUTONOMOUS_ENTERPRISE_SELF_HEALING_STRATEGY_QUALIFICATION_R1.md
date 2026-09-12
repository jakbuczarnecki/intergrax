# Autonomous Enterprise Self-Healing Strategy — Qualification (R1)

**Task:** `AUTONOMOUS-ENTERPRISE-SELF-HEALING-STRATEGY-R1`

**Suite:** `tests/unit/runtime/self_healing/test_autonomous_enterprise_self_healing_strategy_r1_q.py`

| Area | Test | Requirement |
| ---- | ---- | ----------- |
| Plugin contract | `test_custom_strategy_implements_platform_contract` | Application plugin implements `SelfHealingStrategy`; no execute surface |
| Default strategy | `test_platform_default_strategy_available` | Platform defaults registered and resolvable |
| Override | `test_application_strategy_overrides_default` | Higher specificity / tenant scope wins resolution |
| Isolation | `test_failed_strategy_does_not_break_engine` | `STRATEGY_FAILED`; fallback strategy continues |
| Governance | `test_strategy_cannot_bypass_governance` | Denied when governance not approved |
| Governance | `test_high_confidence_still_requires_governance_when_marked` | Production / approval path enforced |
| Execution | `test_strategy_uses_external_operation_spine` | Admitted attempt uses `ext_op_intent_*` |
| Diagnostics | `test_failed_self_healing_creates_central_evidence` | `EXTERNAL_OPERATION_FAILED` → assessment finding |
| Tenant | `test_strategy_is_tenant_scoped` | Registry filters by tenant scope |
| Security | `test_strategy_cannot_emit_execution_directly` | Decision/strategy without execute(); provider not called |
| Read model | `test_self_healing_history_projects_to_investigation_read_model` | Audit → `RelatedSelfHealingHistoryEntryView` |

**Pass criteria:** all tests green; no prohibited execution or diagnostic bypass paths in self-healing tier.
