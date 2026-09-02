# intergrax_assistant agent

UAEP-first scaffold. Full process: [`../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md) (single canonical guide).

## Docs

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) - purpose, contracts, runtime layout
- [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) - task queue and verification
- [`adr/README.md`](docs/adr/README.md) - architecture decision records

## Quick start

1. Implement domain logic in `steps/`
2. Run smoke test: `uv run pytest agents/intergrax_assistant/tests -q`
3. For lab HTTP: register in `applications/lab_application/host/wiring.py` (see guide Step 4C)

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax_assistant.intergrax_assistant_agent import IntergraxAssistantAgent

registry = AgentRegistry()
registry.register(IntergraxAssistantAgent())
```

See **Step 4** in guides/AGENT_CREATION_GUIDE.md for all registration contexts.

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
