# external_contractor_adapter agent

Typed **reflex** cognitive agent — standalone smoke tests under ``agents/external_contractor_adapter/tests/``.

**Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Plan:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · **ADRs:** [`docs/adr/README.md`](docs/adr/README.md)

Full process: [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md)

## Standalone verification

From repository root:

```bash
uv run pytest agents/external_contractor_adapter/tests -q
```

Stub LLM in ``external_contractor_adapter_agent.py`` keeps tests offline — no Tier-3 host required.

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from external_contractor_adapter.external_contractor_adapter_agent import ExternalContractorAdapterAgent

registry = AgentRegistry()
registry.register(ExternalContractorAdapterAgent())
```

See **Step 4** in ``docs/guides/AGENT_CREATION_GUIDE.md`` for host wiring.

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
