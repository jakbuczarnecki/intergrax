# PLAN T-IGX-036-T — PDC-P6 integration repairs

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P6-R**) and the integration findings recorded on the PDC board and in the lane reports (B-D2, B-D3, B-D4, B-D7, B-D8, B-REG-P). Companion of ticket T-IGX-036-S (cross-domain tests). Requirements: no new requirement rows; this batch closes integration defects that the P5 lanes could not fix inside their write sets.

All P5 batches are merged on `development` and MCP-SERVER-X is closed, so the OD-PDC-SEQ-01 gate is met. A merge of this batch changes product paths; the IIAC activation is already invalidated and PDC-P7-00 re-activates it.

## Deliverables

1. **Windows timezone data (D7).** Declare `tzdata; sys_platform == 'win32'` as a direct dependency in `pyproject.toml` (and refresh `uv.lock` with a minimal `uv lock`), because `intergrax/runtime/long_running/recurrence.py` uses IANA zones (`Europe/Warsaw` and others) and Windows has no system zone database. If `uv lock` changes unrelated pins, stop and report instead of committing them. Test: a zone lookup for an IANA name works in a default environment.
2. **D3 capability flags through the evidence adapter.** `model_call_runtime_evidence_adapter.py` delegates `supports_tool_streaming` (and any other `supports_*` flag the D3 conformance gate declares) to the wrapped adapter, so the wrapper does not hide a declared capability. Test: the wrapper reports what the wrapped adapter declares; the D3 conformance gate stays green.
3. **D2 freshness through multi-pass retrieval.** `agentic_loop.py` carries the typed `freshness` verdict of the underlying retrieval result into the loop's final result when the single-pass service produced one, and keeps `None` when it did not. Additive; behavior without an observer is unchanged. Test: a multi-pass retrieval with a stale source returns the verdict.
4. **D4 launcher path.** `resolve_lkw_hybrid_daemon_wiring` probes `scripts/maintenance/lkw-host.py`, which is where the launcher lives, instead of `scripts/lkw-host.py`. Test: the resolver finds the real launcher; `test_audit_ideal_28_3_lkw_hybrid_daemon` is re-run and its result recorded.
5. **Observability circular import.** `register_default_integrations` as the first import no longer raises the circular `ImportError` in `runtime/integrations/observability.py` (`cannot import name 'ObservabilityVendorIntegrationContract'`). The smallest fix (a deferred import or a corrected import order) inside the files of this write set. Test: a subprocess that imports `register_default_integrations` first succeeds; `tests/unit/test_dep1_dependency_foundation.py::test_default_embedding_bootstrap_imports_provider_only_on_selection` is re-run and its result recorded.
6. **D8 trace alias callers.** The seven modules that still import the old trace `ArtifactRef` alias move to `TraceArtifactRef`. The COMPAT-X shim from B-D8 stays in place for external callers. Test: none of the seven imports the alias and a gate fails if an in-repo module reintroduces it.
7. **D8 feature docs registration.** Register the `ARTIFACT_LIFECYCLE` feature pair in `docs/project/capabilities/README.md` and in the runtime hub `docs/project/architecture/intergrax_runtime_architecture.md`, so the docs-pair checker counts it. `scripts/docs/check_docs_domain_pairs.py` is run and its result recorded.
8. **Finding-driven items.** The XD runs of ticket T-IGX-036-S are expected to find further integration defects (for example D8 composition-root wiring of `ToolOutputSpiller`, the governed read port and the retention sweep; strict hosts passing `plugin_provenance` and calling `assert_strict_security_configuration`). Each such item enters this batch only through a write-set amendment under `DELEGATION-PDC-P6-REPAIR.md`, with its exact files.

Design constraints: smallest change per item; additive where a public symbol is involved; no new registry, scheduler, engine, trust, policy or audit authority; no weakening of any gate (a gate may be strengthened or given a classification entry).

## Write set

`pyproject.toml`, `uv.lock`, `intergrax/runtime/llm/model_call_runtime_evidence_adapter.py`, `intergrax/rag/retrieval/agentic_loop.py`, `intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py`, `intergrax/runtime/integrations/observability.py`, `intergrax/runtime/execution/agent_runtime_context.py`, `intergrax/runtime/execution/trace_persistence_models_bridge.py`, `intergrax/runtime/nexus/context/context_manager.py`, `intergrax/runtime/nexus/context/shared_task_context.py`, `intergrax/runtime/nexus/engine/runtime_state.py`, `intergrax/runtime/nexus/engine/contracts/runtime_state_contract.py`, `intergrax/runtime/nexus/tracing/persistence_models.py`, `docs/project/capabilities/README.md`, `docs/project/architecture/intergrax_runtime_architecture.md`, `tests/qualification/pdc_x/p6_repair/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who decides between an amendment under `DELEGATION-PDC-P6-REPAIR.md` and a return to the Operator.

## Protocol

1. Write the repair tests first, under `tests/qualification/pdc_x/p6_repair/`.
2. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files. Do not pass `-p no:cacheprovider`.
3. Commit with `Ref: governance/tickets/T-IGX-036-T.md`, push to a branch and open a PR against `development`. The orchestrator merges it under `DELEGATION-PDC-STANDING.md` authorization 3 (known red classes only: IIAC activation).
4. Record the implementation SHA; the publication SHA is recorded after the merge.
5. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Each of the seven fixed items has a passing test at the implementation SHA, and its named existing test or gate result is recorded.
- Layer boundaries hold; no new authority; no weakened gate.
- The full `tests/qualification/pdc_x` suite still passes (677 passed, 5 skipped before this batch).
- Test log under `.tmp/proof/pdc-x-p5/B-P6-R/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_repair -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

The IIAC protocol header and activation (PDC-P7-00); the compat_x baseline counters that were red before PDC; the unused `embedding_provider/ollama/manifest.py`; Celery, Kafka and RabbitMQ identity and dead-letter mapping (IA-D7 point); FRZ changes; OD signatures; certification.
