# DisputeAnalystAgent

Argument analysis for **Dispute Simulation Workspace (DSW)** - strength/weakness matrix, evidence gaps, party positions.

**Host:** [`applications/dispute_sim_application/`](../../applications/dispute_sim_application/) · **Roster:** [`agents/README.md`](../README.md)

## Docs

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · [`adr/README.md`](docs/adr/README.md)

## Quick start

1. Implement domain logic in `steps/`
2. `uv run pytest agents/dispute_analyst/tests -q`
3. Host run: `capability: dispute.analyze` on `POST /v1/dispute_sim/run` (port 8025)

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from dispute_analyst.dispute_analyst_agent import DisputeAnalystAgent

registry = AgentRegistry()
registry.register(DisputeAnalystAgent())
```

See **Step 4** in guides/AGENT_CREATION_GUIDE.md for all registration contexts.

## Capabilities

`dispute.analyze`

## Layout

- ``dispute_analyst_agent.py`` - Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` - AgentContract
- ``steps/`` - domain execution
- ``prompts/`` - prompt assets
- ``schemas/`` - I/O models
- ``tests/`` - agent smoke tests
- ``notebooks/`` - interactive experiments
- ``docs`` - architecture, plan, ADRs, journal
