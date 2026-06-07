# signoff_probe agent

Architecture: [ARCHITECTURE.md](ARCHITECTURE.md) · Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Full process: [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md).

## Quick start

1. Implement domain logic in `steps/`
2. Run smoke test: `uv run pytest agents/signoff_probe/tests -q`
3. For lab HTTP: register in `applications/lab_application/host/wiring.py` (see guide Step 4C)

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from signoff_probe.signoff_probe_agent import SignoffProbeAgent

registry = AgentRegistry()
registry.register(SignoffProbeAgent())
```

See **Step 4** in AGENT_CREATION_GUIDE.md for all registration contexts.

## Capabilities

`signoff.probe`

## Layout

- ``signoff_probe_agent.py`` — Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` — AgentContract
- ``steps/`` — domain execution
- ``prompts/`` — prompt assets
- ``schemas/`` — I/O models
- ``tests/`` — agent smoke tests
- ``notebooks/`` — interactive experiments
