# PLAN T-IGX-036-M — PDC-P5 batch B-REG-P: duplicate `ollama` catalog slug

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-REG-P**) and `PDC_REG_DIAGNOSIS.md` (row PDC-D1-R022, verdict REGRESSION). No ADR: this is an integration correction. Requirements: PDC-D1-R022.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). Implementation in a worktree is allowed (Operator, 2026-11-13, `DELEGATION-PDC-STANDING.md`); only the merge waits. A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Root cause.** Commit `7aa9cfa18` (EBH-7 gap closure) made `embedding_provider/ollama` register a second catalog row with the slug `ollama`, which `model_serving_runtime/ollama` already registers. `register_default_integrations(preset="full")` then raises `ValueError: Integration slug 'ollama' is already registered`, so every default `create_default_embedding_manager()` fails on a fresh catalog.
2. **Fix.** Make the embedding provider contribute to the single existing `ollama` catalog row (restore the augment registration, or merge into that row) instead of registering a second row. The catalog keeps exactly one `ollama` slug; capability metadata for embeddings is added to that row. The choice is the smaller change that satisfies the catalog contract and is recorded in the checkpoint.
3. **Tests first.** A regression test under `tests/qualification/pdc_x/reg_p/` that builds the full preset on a fresh catalog and asserts: no duplicate slug, one `ollama` row that carries both the model-serving and the embedding contribution, and `create_default_embedding_manager()` succeeds. The existing `tests/unit/rag/test_rag_plugin_discovery.py` and `tests/qualification/plug_03` must be green.
4. **Circular import note.** The diagnosis recorded a separate circular-import `ImportError` in `runtime/integrations/observability.py` when `register_default_integrations` is the first import. It is not part of this batch: the implementer checks whether it still reproduces, reports the result, and does not fix it outside the write set.

Design constraints: no new catalog, registry or authority; the existing integration catalog stays the single owner of slugs. Public catalog contracts do not change. Other embedding providers are not touched.

## Write set

`intergrax/integrations/providers/embedding_provider/`, `tests/qualification/pdc_x/reg_p/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator. The test directory `tests/qualification/pdc_x/reg_p/` was added to the WBS write set together with this plan (tests only).

## Protocol

1. Write the regression test first, under `tests/qualification/pdc_x/reg_p/`.
2. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files. Do not pass `-p no:cacheprovider`.
3. Commit with `Ref: governance/tickets/T-IGX-036-M.md`, push to a branch, open a PR against `development`, and **do not merge**.
4. Record the implementation SHA; the publication SHA is recorded after the merge.
5. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- The regression test fails before the fix and passes after it.
- `tests/unit/rag/test_rag_plugin_discovery.py` and `tests/qualification/plug_03` are green.
- Layer boundaries hold; no new registry or authority; no weakened gate.
- Test log under `.tmp/proof/pdc-x-p5/B-REG-P/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/reg_p tests/unit/rag/test_rag_plugin_discovery.py tests/qualification/plug_03 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Other batches; the circular import in `runtime/integrations/observability.py` (reported only); FRZ changes; OD signatures; merges; IIAC activation.
