# boundary_demo agent

UAEP demo agent for **Execution Boundary Export** - single step calling `records.put` (`attestation.demo` capability).

**Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Plan:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · **ADRs:** [`docs/adr/README.md`](docs/adr/README.md)

**Host:** [`applications/attestation_demo/`](../../applications/attestation_demo/)

## Standalone verification

From repository root:

```bash
uv run pytest agents/boundary_demo/tests -q
```

Full tool-resolution regression (requires attestation host wiring):

```bash
uv run pytest tests/unit/agents/test_boundary_demo_skill_resolution.py -q
```

## Unit-test authoring (isolated)

```python
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from boundary_demo.boundary_demo_agent import BoundaryDemoAgent

agent = BoundaryDemoAgent()
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

Production PoC mounts the agent via ``applications/attestation_demo/host/agent_builders.py`` and Agent Distribution lifecycle.

## Capabilities

`attestation.demo`

## Layout

- ``boundary_demo_agent.py`` - UAEP agent (`get_steps` / `run_step`)
- ``capabilities.py`` - capability ids
- ``tests/`` - contract smoke tests
- ``docs`` - architecture, plan, ADRs, journal
