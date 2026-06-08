# DisputeIntakeAgent

Case material intake for **Dispute Simulation Workspace (DSW)** — classify documents, build chronology, ingest to case-scoped RAG.

**Host:** [`applications/dispute_sim_application/`](../../applications/dispute_sim_application/) · **Product architecture:** [ARCHITECTURE.md](../../applications/dispute_sim_application/ARCHITECTURE.md)  
**Agent roster:** [`agents/README.md`](../README.md)

## Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — purpose, contracts, runtime layout
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — task queue and verification
- [`adr/README.md`](adr/README.md) — architecture decision records

## Quick start

1. Implement domain logic in `steps/`
2. Run smoke test: `uv run pytest agents/dispute_intake/tests -q`
3. Run via product host: `uv run uvicorn dispute_sim_application.host.main:app --port 8025` → `POST /v1/dispute_sim/run` with `capability: dispute.intake`

## Register (programmatic)

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from dispute_intake.dispute_intake_agent import DisputeIntakeAgent

registry = AgentRegistry()
registry.register(DisputeIntakeAgent())
```

See **Step 4** in guides/AGENT_CREATION_GUIDE.md for all registration contexts.

## Capabilities

`dispute.intake`

## Layout

- ``dispute_intake_agent.py`` — Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` — AgentContract
- ``steps/`` — domain execution
- ``prompts/`` — prompt assets
- ``schemas/`` — I/O models
- ``tests/`` — agent smoke tests
- ``notebooks/`` — interactive experiments
- ``adr/`` — architecture decision records
