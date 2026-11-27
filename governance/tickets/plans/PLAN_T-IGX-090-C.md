# Plan: T-IGX-090-C — Wave 2 Test-Only Guard Hardening (`OD-037-GUARD`, `EDER-F-0091`, `EDER-F-0236`)

## Objective

Lock three verified current-HEAD invariants with strictly test-only unit tests (zero edits to `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/`, or `docs/`):

1. **`OD-037-GUARD` (`EQOA-A4-037`):** Lock existing `NexusLoop` `production_mode` human-decision persistence guard (`nexus_loop.py:334`) and `persist_human_decision` return-without-writing contract (`human_response.py:91`) without altering production behavior.
2. **`EDER-F-0091` (`EQOA-A3-011`, `FRZ-TYP-04`):** Add an AST ratchet regression gate over `intergrax/rag/vectorstore/` pinning the exact 4 residual `hasattr(...)` sites on current HEAD (`vector_store.py`, `lexical_hybrid.py` x2, `prod_slo.py`) so zero new unclassified `hasattr()` probes can be introduced, plus behavioral unit tests exercising `VectorStore.list_collections`, `LexicalHybridSupport.query_hybrid`, and `run_beta_adapter_soak`.
3. **`EDER-F-0236` (`EQOA-A8-020`):** Add unit tests covering `from_legacy_rag_hit(..., bound_tenant_id=...)` fail-closed tenant binding (`intergrax/rag/document_loaders/compat/legacy_runtime_document.py`) and the current-HEAD `CompressionReceipt` / `_derive_receipt_id` tenant-attribution and receipt-identity contract (`intergrax/runtime/token_optimization/receipts.py`).

## Authorization Basis

- Operator Decision Record `OP-EC-01` (2026-11-26T17:26:01+01:00): `OD-037-GUARD=A`, `OD-W2-DOCS=A`.
- Wave 2 `NO_NAMED_TEST` 34-obligation binding map (`NEEDS_NEW_UNIT_TEST` = `EDER-F-0091`, `EDER-F-0236`).

## Edit Scope (Strict)

- `governance/tickets/plans/PLAN_T-IGX-090-C.md`
- `governance/tickets/T-IGX-090-C.md`
- `tests/unit/runtime/nexus/test_od_037_guard_human_decision_persistence.py`
- `tests/unit/rag/vectorstore/test_frz_typ04_vectorstore_reflection_gate.py`
- `tests/unit/rag/document_loaders/test_legacy_runtime_document_tenant_scope.py`

## Out of Scope

- Any edits under `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/`, or `docs/`.
- Changing `persist_human_decision` (`OD-037-PERSIST` was rejected under `OP-EC-01`).
- Removing residual `hasattr()` sites or modifying `_derive_receipt_id` product code.
- Any `CLOSED`, `IIAC PASS`, `FRZ PASS`, or `G6 PASS` claim, or any `git push` / merge to `development`.

## Verification Contract (Serial)

1. `uv run pytest tests/unit/runtime/nexus/test_od_037_guard_human_decision_persistence.py tests/unit/rag/vectorstore/test_frz_typ04_vectorstore_reflection_gate.py tests/unit/rag/document_loaders/test_legacy_runtime_document_tenant_scope.py -q`
2. `uv run ruff check tests/unit/runtime/nexus/test_od_037_guard_human_decision_persistence.py tests/unit/rag/vectorstore/test_frz_typ04_vectorstore_reflection_gate.py tests/unit/rag/document_loaders/test_legacy_runtime_document_tenant_scope.py`
3. `uv run python -m scripts.governance.verify_plan_approval governance/tickets/T-IGX-090-C.md --plan governance/tickets/plans/PLAN_T-IGX-090-C.md`
4. `uv run python -m scripts.governance.verify_ticket_lifecycle governance/tickets/T-IGX-090-C.md --plan governance/tickets/plans/PLAN_T-IGX-090-C.md`
5. `uv run python -m scripts.governance.verify_ticket_approvals --ticket governance/tickets/T-IGX-090-C.md`
