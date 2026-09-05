# DisputeIntakeAgent

Case material intake for **Dispute Simulation Workspace (DSW)** - classify documents, build chronology, ingest to case-scoped RAG.

**Host:** [`applications/dispute_sim_application/`](../../applications/dispute_sim_application/) · **Product architecture:** [ARCHITECTURE.md](../../applications/dispute_sim_application/docs/ARCHITECTURE.md)
**Agent roster:** [`agents/README.md`](../README.md)

## Docs

- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) - purpose, contracts, runtime layout
- [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) - task queue and verification
- [`adr/README.md`](docs/adr/README.md) - architecture decision records

## Quick start

1. Implement domain logic in `steps/`
2. Run smoke test: `uv run pytest agents/dispute_intake/tests -q`
3. Run via product host: `uv run uvicorn dispute_sim_application.host.main:app --port 8025` → `POST /v1/dispute_sim/run` with `capability: dispute.intake`

## Unit-test authoring (isolated)

```python
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from dispute_intake.dispute_intake_agent import DisputeIntakeAgent

agent = DisputeIntakeAgent()
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


## ## Layout

- ``dispute_intake_agent.py`` - Agent class (UAEP)
- ``contract.py`` / ``capabilities.py`` - AgentContract
- ``steps/`` - domain execution
- ``prompts/`` - prompt assets
- ``schemas/`` - I/O models
- ``tests/`` - agent smoke tests
- ``notebooks/`` - interactive experiments
- ``docs`` - architecture, plan, ADRs, journal
