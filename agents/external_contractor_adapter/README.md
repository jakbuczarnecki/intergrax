# external_contractor_adapter agent

Typed **reflex** Tier-2 external-work adapter (GEC-3) — maps via injected ``ExternalWorkIntegration``.

**Architecture:** [`docs/ARCHITECTURE.md`](../../docs/project/technical/agents/external_contractor_adapter/ARCHITECTURE.md) · **Plan:** [`docs/IMPLEMENTATION_PLAN.md`](../../docs/project/technical/agents/external_contractor_adapter/IMPLEMENTATION_PLAN.md) · **ADRs:** [`docs/project/technical/adr/README.md`](../../docs/project/technical/agents/external_contractor_adapter/adr/README.md)

Full process: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md)

## Standalone verification

From repository root:

```bash
uv run pytest agents/external_contractor_adapter/tests -q
```

Stub LLM + ``tests/fakes/DeterministicExternalWorkFake`` keep tests offline — no network / Tier-3 host required.

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from external_contractor_adapter.external_contractor_adapter_agent import ExternalContractorAdapterAgent

registry = AgentRegistry()
registry.register(
    ExternalContractorAdapterAgent(
        external_work=my_integration,
        side_effect_policy=my_side_effect_policy,
    )
)
```

See **Step 4** in ``docs/project/technical/guides/AGENT_CREATION_GUIDE.md`` for host wiring. Host may inject via ``settings.external_work_integration`` and ``settings.meaningful_side_effect_policy``.

## Capabilities

`external_contractor.adapt`

## Layout

- ``external_contractor_adapter_agent.py`` — Agent class (ACP hooks)
- ``contract.py`` / ``capabilities.py`` — AgentContract
- ``steps/`` — domain execution
- ``prompts/`` — prompt assets
- ``schemas/`` — I/O models
- ``tracing/`` — DiagnosticPayload extensions
- ``signals/`` — domain signal payloads
- ``tests/`` — standalone agent smoke tests
- ``notebooks/`` — interactive experiments
- ``docs/`` — architecture, plan, ADRs, journal
