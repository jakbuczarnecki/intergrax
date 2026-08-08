# boundary_demo agent — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

**Status:** Stable PoC baseline — legacy UAEP (not ACP scaffold).

---

## Purpose

Tier-2 **partner sandbox** agent for Execution Boundary Export (EBE). One step stores a demo record via Harness tool gateway (`records.put`). No attestation receipt logic in the agent.

## Capabilities

- `attestation.demo`

## Layout

| Path | Role |
|------|------|
| `boundary_demo_agent.py` | `BoundaryDemoAgent` — UAEP `get_steps` / `run_step` |
| `capabilities.py` | Capability ids |
| `tests` | Standalone contract smoke tests |
| `docs/project/technical/adr` | Agent ADRs — [`adr/README.md`](adr/README.md) |

## Runtime

- Legacy **UAEP** pipeline (`Agent` + `run_step`), not ACP `perceive/reason/act`
- `LabHarnessContext` + optional `ToolEnablementProfile` / boundary export settings injected by Tier-3 host
- Tool: `records.put` (resolved via registry + skill `data.records_admin`, not predeclared on author contract)
- Stub LLM: `PrefixStubLLMAdapter(prefix="boundary-demo")`

## Behavior

```text
run_step(store_demo_record):
  1. Read partition_key, row_key, record_data from request metadata (defaults for PoC)
  2. ctx.invoke_tool(ToolRequest(tool_name="records.put", ...))
  3. Return StepOutput with stored partition/row or error
```

## Contract

| Field | Value |
|-------|-------|
| `agent_id` | `boundary_demo_agent` |
| `capabilities` | `attestation.demo` |
| `skills` | `data.records_admin` |
| `allowed_tools` (author) | `[]` — runtime merge via registry |
| `max_steps` | 1 |

## Tier hygiene

- Imports only `intergrax.*` and `boundary_demo` — **no** `applications` imports
- Tool wiring supplied by `applications/attestation_demo/host/tool_wiring.py`

## Registration

- Tier-3 host: `AgentBinding.mount(BoundaryDemoAgent, ...)` in `applications/attestation_demo/manifest.py`
- Canon: [`docs/project/technical/applications/attestation_demo/ARCHITECTURE.md`](../../applications/attestation_demo/ARCHITECTURE.md) §10
