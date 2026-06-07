# dispute_intake agent

UAEP-first scaffold. Full process: [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md) (single canonical guide).

## Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — purpose, contracts, runtime layout
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — task queue and verification
- [`adr/README.md`](adr/README.md) — architecture decision records

## Quick start

1. Implement domain logic in `steps/`
2. Run smoke test: `uv run pytest agents/dispute_intake/tests -q`
3. For lab HTTP: register in `applications/lab_application/host/wiring.py` (see guide Step 4C)

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from dispute_intake.dispute_intake_agent import DisputeIntakeAgent

registry = AgentRegistry()
registry.register(DisputeIntakeAgent())
```

See **Step 4** in AGENT_CREATION_GUIDE.md for all registration contexts.

## Capabilities

`dispute.intake`

## Layout

- ``dispute_intake_agent.py`` — Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` — AgentContract
- ``steps/`` — domain execution
- ``prompts/`` — prompt assets
- ``schemas/`` — I/O models
- ``tests/`` — agent smoke tests
- ``notebooks/`` — interactive experiments
- ``adr/`` — architecture decision records
