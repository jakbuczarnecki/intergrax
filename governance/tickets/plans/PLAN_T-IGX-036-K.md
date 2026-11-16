# PLAN T-IGX-036-K — PDC-P5 batch B-D3: D3 Model Runtime

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-D3**: declared streaming/tool-streaming/cancel capabilities, conformance gate, catalog-owned context windows). Decision record: ADR-PDC-03 (Accepted). Requirements: PDC-D3-R004, R006, R009, R011, R015, R022, R023.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Declared capabilities (R006, R022).** Extend the existing `supports_*` flag family with `supports_tool_streaming`, and expose the cancellation seam outcome (`terminated`, `transport_closed`, `not_supported`) as a typed capability read through `LLMAdapterRegistry`.
2. **Conformance gate (R022).** A gate asserts each flag matches the implemented methods: no adapter streams while reporting `False`. Fix the six adapters that implement `stream_messages` but report `False`.
3. **Fail closed (R006, R015).** The runtime checks requested features against declared capabilities before the call and fails with a typed error. It never silently downgrades streamed tool calls to buffered output.
4. **Cancel (R011), refusal (R009), failover (R004).** Typed cancel outcome per provider (fake providers), refusal normalized across providers, bounded failover exhaustion.
5. **Context window (R023).** `ModelCatalog` stays the single source; provider-local tables become fallbacks resolved through the catalog.

Design constraints: Capabilities are declared, not forced: no streaming parity is required of every provider. No second capability registry outside `LLMAdapterRegistry` / `ModelCatalog`. The ABI that agents see is unchanged. The LLM provider plugin layer is out of scope (OD-PDC-SCOPE-03 = A).

## Write set

`intergrax/llm_adapters/`, `tests/qualification/pdc_x/d3/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who is the single integration owner for the shared hotspots (`intergrax/applications/_shared/`, `intergrax/contracts/`).

## Protocol

1. Reuse the canonical mechanism named in ADR-PDC-03.
2. Write the tests first, under `tests/qualification/pdc_x/d3/`.
3. One `uv run pytest` per verification step; ruff and pyright on touched files.
4. Commit with `Ref: governance/tickets/T-IGX-036-K.md`, push to a branch, open a PR against `development`, and **do not merge**.
5. Record the implementation SHA; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Every requirement row of the batch has a passing test at the implementation SHA.
- Layer boundaries hold (`intergrax/` does not import `agents/` or `applications/`).
- No new registry, scheduler or authority beside the existing owner.
- Existing suites for the touched packages stay green (no weakened gate).
- Test log under `.tmp/proof/pdc-x-p5/B-D3/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d3 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

The implementer adds one more bounded `uv run pytest` over the existing unit suites of the touched packages and records it in the checkpoint.

## Out of scope

Other batches; the `ollama` regression (B-REG-P); FRZ changes; OD signatures; merges; IIAC activation.
