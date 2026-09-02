Before submitting substantial work, read [docs/project/community/COLLABORATION.md](../docs/project/community/COLLABORATION.md) and [LICENSE](../LICENSE).

## Summary

<!-- What changed and why? How does this align with the Harness AI strategic goal? -->

## Type of change

- [ ] Harness maintenance (§6.1)
- [ ] Bug fix
- [ ] Documentation (docs/ canon update)
- [ ] Tier-2 agent
- [ ] Tier-0 plugin (integration / tool / skill)
- [ ] Tier-3 application
- [ ] Nexus runtime / harness architecture
- [ ] CI / tooling
- [ ] Other (describe below)

## Documentation updated

<!-- One source of truth per topic - list files updated in docs/ -->

- [ ] No documentation changes needed
- [ ] docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md
- [ ] docs/project/architecture/intergrax_runtime_architecture.md
- [ ] docs/project/technical/guides/AGENT_CREATION_GUIDE.md
- [ ] Other: <!-- list -->

## Harness layer checklist (IDEAL-32.1)

<!-- Mark affected AUDIT_MAP layers (1–32); link IDEAL-* ID when applicable -->

- [ ] Layer impact assessed (see `docs/audit_results/AUDIT_PROTOCOL.md`)
- [ ] Domain plan pair updated when contracts change (`docs/project/architecture/<DOMAIN>.md` + `docs/project/maintainers/plans/<DOMAIN>.md`)

## Architecture compliance

- [ ] Respects tier dependency boundaries (`intergrax/` ↛ `agents/`/`applications/`)
- [ ] Reuses existing Tier-0 modules (no parallel universal mechanisms)
- [ ] Does not modify `intergrax/runtime/` for agent-specific needs (if agent PR)
- [ ] Aligns with [INTERGRAX_DEVELOPMENT_STRATEGY.md](../docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

## Test evidence

```bash
# Paste command output
uv run pytest -m gate -q
```

- [ ] Gate tests pass
- [ ] `python scripts/maintenance/check_harness_no_getattr.py` (if harness touched)
- [ ] `python scripts/maintenance/check_harness_guardrail_wiring.py` (if guardrails / integrations touched)
- [ ] Agent tests pass (if agent touched)

## Phase / task reference

<!-- Link to implementation plan section if applicable, e.g. §6.1, Phase W-ADAPT, AS-1 -->

## Checklist

- [ ] Minimal scope - no unrelated changes
- [ ] No secrets committed
- [ ] Copyright header on new files
- [ ] Follows [CONTRIBUTING.md](../CONTRIBUTING.md) work cycle
- [ ] Read [docs/project/community/COLLABORATION.md](../docs/project/community/COLLABORATION.md) and [LICENSE](../LICENSE) for substantial changes
