# legal agent

UAEP-first scaffold. Full process: [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md) (single canonical guide).

## Quick start

1. Implement domain logic in `steps/`
2. Run smoke test: `uv run pytest agents/legal/tests -q`
3. For lab HTTP: register in `applications/lab_application/host/wiring.py` (see guide Step 4C)

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from legal.legal_agent import LegalAgent

registry = AgentRegistry()
registry.register(LegalAgent())
```

See **Step 4** in AGENT_CREATION_GUIDE.md for all registration contexts.

## Capabilities

`legal.review`

## Layout

- ``legal_agent.py`` — Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` — AgentContract
- ``steps/`` — domain execution
- ``prompts/`` — prompt assets
- ``schemas/`` — I/O models
- ``tests/`` — agent smoke tests
- ``notebooks/`` — interactive experiments
