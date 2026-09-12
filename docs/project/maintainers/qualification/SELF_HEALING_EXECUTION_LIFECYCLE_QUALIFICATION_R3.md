# Self-healing execution lifecycle qualification (R3)

## Prerequisite

`SELF_HEALING_R2_IMPLEMENTATION_AUDIT` — all dimensions PASS, verdict APPROVED.

## Matrix

| Test | Requirement |
|------|-------------|
| `test_healing_workflow_full_lifecycle` | R3 lifecycle completes with correlation |
| `test_healing_never_executes_directly` | Lifecycle engine has no direct spine gate |
| `test_execution_id_is_preserved` | Correlation IDs ⊆ spine audit |
| `test_validation_requires_observation_evidence` | Pipeline rejects missing observation |
| `test_failed_validation_triggers_rollback` | Rollback lifecycle terminus |
| `test_custom_validator_plugin` | Validator SPI |
| `test_custom_strategy_selector` | Selector SPI |
| `test_strategy_quality_feedback` | Performance feedback + selector |
| `test_plugin_failure_containment` | `PLUGIN_FAILED` boundary |
| `test_cross_tenant_healing_isolation` | Tenant mismatch rejected |
| `test_healing_execution_timeline_projection` | Read model timeline |

## Command

```bash
uv run pytest tests/unit/runtime/self_healing/test_autonomous_enterprise_self_healing_orchestration_r3_q.py tests/unit/runtime/self_healing/test_autonomous_enterprise_self_healing_orchestration_r2_q.py -q
```

## Frozen boundaries

Lifecycle states, execution correlation, observation-before-validation, rollback via spine, selector SPI, learning metrics, read-only investigation timeline.
