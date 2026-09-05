# LegalAgent

Contract review capability - distinct from **DSW** dispute lifecycle agents.

**Host:** [`applications/legal_application/`](../../applications/legal_application/) · **Roster:** [`agents/README.md`](../README.md)  
Architecture: [ARCHITECTURE.md](docs/ARCHITECTURE.md) · Plan: [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) · Guide: [`../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md)

## Quick start

1. Implement domain logic in `steps/`
2. Run smoke test: `uv run pytest agents/legal/tests -q`
3. For lab HTTP: add `AgentBinding.mount(...)` in lab manifest (see guide Step 4C)

## Unit-test authoring (isolated)

```python
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from legal.legal_agent import LegalAgent

agent = LegalAgent()
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

See **Step 4** in ``docs/project/technical/guides/AGENT_CREATION_GUIDE.md`` for all integration contexts.



## ## Layout

- ``legal_agent.py`` - Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` - AgentContract
- ``steps/`` - domain execution
- ``prompts/`` - prompt assets
- ``schemas/`` - I/O models
- ``tests/`` - agent smoke tests
- ``notebooks/`` - interactive experiments
