# Self-healing adaptive intelligence qualification (R4)

## Prerequisite

R3 execution lifecycle qualification — PASS.

## Matrix

| Test | Requirement |
|------|-------------|
| `test_custom_strategy_ranking_provider` | Custom ranking SPI changes strategy order |
| `test_strategy_failure_isolated` | Plugin exception does not crash engine |
| `test_confidence_requires_evidence` | Zero confidence without evidence |
| `test_adaptive_layer_cannot_execute` | No execution spine imports; selector remains advisory |
| `test_tenant_isolation` | Registry scope + context validation |
| `test_existing_self_healing_regression` | R1–R3 qualification suites PASS |

## Command

```bash
uv run pytest tests/unit/runtime/self_healing/test_autonomous_enterprise_self_healing_adaptive_intelligence_r4_q.py -q
uv run ruff check intergrax/contracts/self_healing/adaptive intergrax/runtime/self_healing/adaptive
uv run pyright intergrax/contracts/self_healing/adaptive intergrax/runtime/self_healing/adaptive
```

## Frozen boundaries

Recommendation-only adaptive layer, SPI registries, plugin isolation statuses, diagnostic read projection, learning repository port.
