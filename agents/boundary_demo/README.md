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

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from boundary_demo.boundary_demo_agent import BoundaryDemoAgent

registry = AgentRegistry()
registry.register(BoundaryDemoAgent())
```

Production PoC mounts the agent via `applications/attestation_demo/host/agent_builders.py`.

## Capabilities

`attestation.demo`

## Layout

- ``boundary_demo_agent.py`` - UAEP agent (`get_steps` / `run_step`)
- ``capabilities.py`` - capability ids
- ``tests/`` - contract smoke tests
- ``docs`` - architecture, plan, ADRs, journal
