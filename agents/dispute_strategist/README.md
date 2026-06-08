# DisputeStrategistAgent

Litigation strategy for **DSW** — attack/defense lines, emphasis map, negotiation posture.

**Host:** [`applications/dispute_sim_application/`](../../applications/dispute_sim_application/) · **Roster:** [`agents/README.md`](../README.md)

## Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) · [`adr/README.md`](adr/README.md)

## Quick start

1. Implement domain logic in `steps/`
2. `uv run pytest agents/dispute_strategist/tests -q`
3. Host run: `capability: dispute.strategy` on `POST /v1/dispute_sim/run` (port 8025)

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from dispute_strategist.dispute_strategist_agent import DisputeStrategistAgent

registry = AgentRegistry()
registry.register(DisputeStrategistAgent())
```

See **Step 4** in guides/AGENT_CREATION_GUIDE.md for all registration contexts.

## Capabilities

`dispute.strategy`

## Layout

- ``dispute_strategist_agent.py`` — Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` — AgentContract
- ``steps/`` — domain execution
- ``prompts/`` — prompt assets
- ``schemas/`` — I/O models
- ``tests/`` — agent smoke tests
- ``notebooks/`` — interactive experiments
- ``adr/`` — architecture decision records
