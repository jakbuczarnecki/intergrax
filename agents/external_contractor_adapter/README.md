# external_contractor_adapter agent

Typed **reflex** Tier-2 external-work adapter (GEC-3) - maps via injected ``ExternalWorkIntegration``.

**Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Plan:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · **ADRs:** [`docs/adr/README.md`](docs/adr/README.md)

Full process: [`../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md)

## Standalone verification

From repository root:

```bash
uv run pytest agents/external_contractor_adapter/tests -q
```

Stub LLM + ``tests/fakes/DeterministicExternalWorkFake`` keep tests offline - no network / Tier-3 host required.

## Unit-test authoring (isolated)

```python
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from external_contractor_adapter.external_contractor_adapter_agent import ExternalContractorAdapterAgent

agent = ExternalContractorAdapterAgent(external_work=my_integration, side_effect_policy=my_policy)
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

Host may inject via ``settings.external_work_integration`` and ``settings.meaningful_side_effect_policy``.

## Capabilities

`external_contractor.adapt`

## Layout

- ``external_contractor_adapter_agent.py`` - Agent class (ACP hooks)
- ``contract.py`` / ``capabilities.py`` - AgentContract
- ``steps/`` - domain execution
- ``prompts/`` - prompt assets
- ``schemas/`` - I/O models
- ``tracing/`` - DiagnosticPayload extensions
- ``signals/`` - domain signal payloads
- ``tests/`` - standalone agent smoke tests
- ``notebooks/`` - interactive experiments
- ``docs`` - architecture, plan, ADRs, journal
