# DisputeStrategistAgent

Litigation strategy for **DSW** - attack/defense lines, emphasis map, negotiation posture.

**Host:** [`applications/dispute_sim_application/`](../../applications/dispute_sim_application/) · **Roster:** [`agents/README.md`](../README.md)

## Docs

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · [`adr/README.md`](docs/adr/README.md)

## Quick start

1. Implement domain logic in `steps/`
2. `uv run pytest agents/dispute_strategist/tests -q`
3. Host run: `capability: dispute.strategy` on `POST /v1/dispute_sim/run` (port 8025)

## Unit-test authoring (isolated)

```python
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from dispute_strategist.dispute_strategist_agent import DisputeStrategistAgent

agent = DisputeStrategistAgent()
result = await agent.run(
    AgentRunRequest(
        input="hello",
        identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        agent_id=agent.contract_id,
    )
)
```

## Lab / product integration

Add the agent via ``AgentBinding.mount(...)`` in the Tier-3 manifest and run through
**Agent Distribution → registry projection → Execution**. Do not use local
``AgentRegistry()`` or ``NexusLoop`` on serving paths.

See **Step 4** in ``docs/project/technical/guides/AGENT_CREATION_GUIDE.md``.


## ## Layout

- ``dispute_strategist_agent.py`` - Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` - AgentContract
- ``steps/`` - domain execution
- ``prompts/`` - prompt assets
- ``schemas/`` - I/O models
- ``tests/`` - agent smoke tests
- ``notebooks/`` - interactive experiments
- ``docs`` - architecture, plan, ADRs, journal
