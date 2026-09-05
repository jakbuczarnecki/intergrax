# intergrax_assistant agent

UAEP-first scaffold. Full process: [`../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md) (single canonical guide).

## Docs

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) - purpose, contracts, runtime layout
- [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) - task queue and verification
- [`adr/README.md`](docs/adr/README.md) - architecture decision records

## Quick start

1. Implement domain logic in `steps/`
2. Run smoke test: `uv run pytest agents/intergrax_assistant/tests -q`
3. For lab HTTP: add `AgentBinding.mount(...)` in lab manifest (see guide Step 4C)

## Unit-test authoring (isolated)

```python
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax_assistant.intergrax_assistant_agent import IntergraxAssistantAgent

agent = IntergraxAssistantAgent()
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

## Capabilities

`platform.assist`

## Layout

- ``intergrax_assistant_agent.py`` - Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` - AgentContract
- ``steps/`` - domain execution
- ``prompts/`` - prompt assets
- ``schemas/`` - I/O models
- ``tests/`` - agent smoke tests
- ``notebooks/`` - interactive experiments
- ``docs`` - architecture, plan, ADRs, journal
