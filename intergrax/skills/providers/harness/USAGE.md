# Harness skill bundle

**Bundle id:** `harness` · **Status:** STABLE · **Plugin:** `HarnessSkillPlugin`

Platform smoke and gate skills — no business domain logic. Enable on every lab host via `harness_platform_skill_profile()` or `lab_skill_profile()`.

## Skills in this bundle

| skill_id | Guide |
|----------|-------|
| `harness.tool_smoke` | [harness.tool_smoke/USAGE.md](harness.tool_smoke/USAGE.md) |
| `harness.context_demo` | [harness.context_demo/USAGE.md](harness.context_demo/USAGE.md) |
| `harness.trace_read` | [harness.trace_read/USAGE.md](harness.trace_read/USAGE.md) |
| `harness.modality_smoke` | [harness.modality_smoke/USAGE.md](harness.modality_smoke/USAGE.md) |
| `harness.vision_qa` | [harness.vision_qa/USAGE.md](harness.vision_qa/USAGE.md) |
| `harness.skill_registry` | [harness.skill_registry/USAGE.md](harness.skill_registry/USAGE.md) |
| `harness.integration_bridge_smoke` | [harness.integration_bridge_smoke/USAGE.md](harness.integration_bridge_smoke/USAGE.md) |
| `harness.reliability_smoke` | [harness.reliability_smoke/USAGE.md](harness.reliability_smoke/USAGE.md) |
| `harness.policy_smoke` | [harness.policy_smoke/USAGE.md](harness.policy_smoke/USAGE.md) |
| `harness.stack_demo` | [harness.stack_demo/USAGE.md](harness.stack_demo/USAGE.md) |

## Registration

```python
from intergrax.skills.providers.harness.bundle import register_harness_skill_bundle
register_harness_skill_bundle()
```
