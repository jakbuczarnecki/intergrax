# PLAN T-IGX-036-N — PDC-P5 batch B-D2: D2 Context & Knowledge

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-D2**: hard-trim fidelity, durable compaction (CE-02 closure), memory retention sweep, RAG freshness, adversarial context). Decision record: ADR-PDC-02 (Accepted). Requirements: PDC-D2-R005, R007, R009, R011, R012, R014, R015.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Hard trim fidelity (R011).** `TOKENIZER_HARD_TRIM` in `degradation_ladder.py` truncates content only and keeps `entry_id`, `name`, `tool_calls` and `tool_call_id`. A tool-call pair that cannot fit is dropped atomically; a call or its result is never orphaned.
2. **Durable compaction (R012).** Adversarial round-trip and reconstruction tests over the existing compaction path. CE-02 is closed by requalifying `tests/qualification/ce_02` at the audit SHA; no new context or compaction engine.
3. **Memory retention (R015).** A typed retention sweep enforces expiry through the existing `MemoryControlPlane` `forget` / `delete` paths and tombstones the matching vector-index entries. The sweep is admitted as background work through canonical execution; no new scheduler and no enforcement inside individual memory providers.
4. **RAG freshness (R005, OD-PDC-SCOPE-06 = A).** Indexed documents carry a source revision or timestamp, retrieval returns a typed staleness verdict, and reindexing is admitted as background work through the existing embedding-version reindex hook. No new retrieval engine. The change stays inside `intergrax/rag/governance/`; if retrieval code outside the write set must change, the batch stops and reports.
5. **Adversarial coverage (R007, R009, R014).** Tests for mandatory instruction retention under token pressure, injected tool output not promoted to an instruction, and a sensitive fragment not sent to the provider. Fixes are made only where a test proves a gap, inside the write set.

Design constraints: The CE-02 ownership table (`CONTEXT_ENGINEERING.md` section 147-165) stays authoritative. No second context assembly or compaction engine (FRZ-OWN). Retrieval-poisoning mounting (PDC-D2-R008) belongs to batch B-D6, not here.

## Write set

`intergrax/context/`, `intergrax/runtime/nexus/context/`, `intergrax/runtime/token_optimization/`, `intergrax/runtime/context_lifecycle/`, `intergrax/memory/`, `intergrax/rag/governance/`, `tests/qualification/pdc_x/d2/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who is the single integration owner for the shared hotspots (`intergrax/applications/_shared/`, `intergrax/contracts/`).

## Protocol

1. Reuse the canonical mechanism named in ADR-PDC-02.
2. Write the tests first, under `tests/qualification/pdc_x/d2/`.
3. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files.
4. Commit with `Ref: governance/tickets/T-IGX-036-N.md`, push to a branch, open a PR against `development`, and **do not merge**.
5. Record the implementation SHA; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Every requirement row of the batch has a passing test at the implementation SHA.
- Layer boundaries hold (`intergrax/` does not import `agents/` or `applications/`).
- No new registry, scheduler, engine or authority beside the existing owner.
- Existing suites for the touched packages stay green (no weakened gate).
- Test log under `.tmp/proof/pdc-x-p5/B-D2/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d2 tests/qualification/ce_02 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

The implementer adds one more bounded `uv run pytest` over the existing unit suites of the touched packages and records it in the checkpoint.

## Out of scope

Other batches (in particular B-D6, which the dependency DAG places after wave 2); the `ollama` regression (B-REG-P); FRZ changes; OD signatures; merges; IIAC activation.
