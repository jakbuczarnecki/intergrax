# T-IGX-041-N — PDC-X batch B-P8-R2: P8 repair loop 1, test hygiene of the closure suites and the incident scenario

## Scope

Test, data and classification repair after the independent global audit IA-GLOBAL (`PDC_INDEPENDENT_AUDIT.md`, pull request `jakbuczarnecki/intergrax#579`, verdict PARTIAL, audited SHA `cba32d2479cd134911d2fdf4fb1db4587c8a94d8`), under decision 2 of `DELEGATION-PDC-CLOSE.md` (one P8 repair loop) and the limits of `DELEGATION-PDC-P7-REPAIR.md` (option B; no hotspot path outside exactly named files; gates only strengthened, classified or re-pinned). The batch writes no product path and no hotspot path. In scope: the test-side residue of G-05 (stale keyword, missing occurrence persistence, an invalid proof descriptor value and a new descriptor gate), G-04 (the `FakeVectorstore` double, one reingest test that queries the coordinator with the wrong key, the wrong CI claim of `RAG_RED_CLASSIFICATION.md`) and G-10 (three stale D5-contract tests and the two `test_audit_ideal_depth_gate.py` reds). Excluded and routed to T-IGX-041-M or an Operator decision: the authority, manifest and registry layers behind the 82 `AgentBinding` failures of the incident scenario (hotspot file `scenario_runtime_baseline.py` and the `ModelCallRuntimeEvidenceAdapter` token count), the two graph reingest tests and `test_graphrag_source_reingest_replaces_vectors_and_graph_safely` (likely product defect in the graph visibility keys), the rebuild-path and trace-schema D5 tests, and the four declared `fixture_strict_adapter_parity` tests. The plan is not a certification verdict.

## Plan digest

Plan digest: 6617cc90dbfc78205323a8e370d02cb09a1c55402429166b400f75ee40d33f54

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-N.md`](plans/PLAN_T-IGX-041-N.md).

## Approved

(pending; the plan is signed by a delegate session under `DELEGATION-PDC-P7-REPAIR.md` or by the Operator; the implementer never writes the approval line)

## Test contract

```text
uv run pytest tests/qualification/pdc_x/n_repair tests/unit/platform_proofs/scenarios/ai_incident_investigation tests/unit/rag/indexing/test_indexing_manager_indexes_documents.py tests/integration/rag/test_same_source_reingest_serialization.py tests/integration/rag/test_graph_reingest_qualification.py tests/unit/applications/test_harness_host_runtime_llm.py tests/unit/applications/test_scenario_runtime_baseline.py tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py -q -p no:randomly
uv run ruff check <touched files>
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Known reds that remain by design and are not repaired here: the excluded rows of the plan (82 incident scenario tests behind the authority and rebuild layers, the three graph reingest tests, the rebuild-path and trace-schema tests of `test_scenario_runtime_baseline.py`), the four `fixture_strict_adapter_parity` tests, IIAC activation, EBH-7 identity.

## Out of scope

- The excluded findings (plan table) and every product or hotspot path (`intergrax/`, `agents/`, `applications/`, `testing_support/`).
- G-01, G-02, G-03, G-06 and G-07..G-12 of the audit (ticket T-IGX-041-M, T-IGX-041-K or the P9 registers).
- `scripts/governance/`, `.github/`, FRZ files and P8 originals.
- IIAC activation and rebind, the EBH-7 re-pin, OD signatures, certification and the re-audit.

## Lifecycle state

plan_pending
