# Self-Healing Orchestration and Learning — Qualification (R2)

**Task:** `AUTONOMOUS-ENTERPRISE-SELF-HEALING-ORCHESTRATION-R2`

**Suite:** `tests/unit/runtime/self_healing/test_autonomous_enterprise_self_healing_orchestration_r2_q.py`

| Area | Test | Requirement |
| ---- | ---- | ----------- |
| Plan | `test_strategy_creates_healing_plan` | Decision → `SelfHealingPlan` |
| Multi-step | `test_multi_step_workflow_execution` | Sequential steps through completion |
| Spine reuse | `test_healing_uses_external_operation_spine` | External operation audit trail |
| Governance | `test_healing_cannot_bypass_governance` | Denied without governance |
| Validation | `test_success_requires_validation` | Success only after validation SPI |
| Rollback | `test_failed_validation_requires_rollback` | Failed validation → rollback path |
| Plugin | `test_custom_validation_plugin` | Custom validation provider |
| Isolation | `test_failed_plugin_is_contained` | `PLUGIN_FAILED` on plugin exception |
| Learning | `test_strategy_quality_updates_from_outcome` | Workflow outcome updates quality |
| Tenant | `test_healing_workflow_tenant_isolation` | Tenant mismatch blocked |
| Read model | `test_workflow_history_projects_to_read_model` | Audit → workflow history view |

**Prerequisite:** R1 qualification suite green.

**Pass criteria:** all R2 tests green; no second execution authority in workflow tier.
