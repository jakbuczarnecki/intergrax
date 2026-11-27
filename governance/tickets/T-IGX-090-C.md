# T-IGX-090-C — Wave 2 Test-Only Guard Hardening (`OD-037-GUARD`, `EDER-F-0091`, `EDER-F-0236`)

## Scope

Strictly test-only guard hardening batch (zero edits to `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/`, or `docs/`) implementing three current-HEAD regression and contract test suites:

1. **`OD-037-GUARD` (`EQOA-A4-037`):** `tests/unit/runtime/nexus/test_od_037_guard_human_decision_persistence.py` — 4 negative/contract tests locking existing `NexusLoop` `production_mode` human-decision persistence guard (`nexus_loop.py:334`) and `persist_human_decision` return-without-writing contract (`human_response.py:91`) without modifying production code.
2. **`EDER-F-0091` (`EQOA-A3-011`, `FRZ-TYP-04`):** `tests/unit/rag/vectorstore/test_frz_typ04_vectorstore_reflection_gate.py` — AST `hasattr()` ratchet gate locking the exact 4 classified residual sites across `intergrax/rag/vectorstore/` (`vector_store.py`, `lexical_hybrid.py` x2, `prod_slo.py`) plus behavioral coverage of `VectorStore.list_collections`, `LexicalHybridSupport.query_hybrid`, and `run_beta_adapter_soak`.
3. **`EDER-F-0236` (`EQOA-A8-020`):** `tests/unit/rag/document_loaders/test_legacy_runtime_document_tenant_scope.py` — unit tests locking fail-closed tenant binding in `from_legacy_rag_hit(..., bound_tenant_id=...)` (`legacy_runtime_document.py`) and the current-HEAD `CompressionReceipt` / `_derive_receipt_id` tenant-attribution and receipt-identity contract (`receipts.py`).

Linked plan: [`governance/tickets/plans/PLAN_T-IGX-090-C.md`](plans/PLAN_T-IGX-090-C.md)

## Plan digest

Plan digest: accae06ac3f6ffc22b4bb09507121627dd71285e5246746931c51247e6668d77

## Approved

Approved: operator @ 2026-11-26T17:26:01+01:00

Decision basis: `OP-EC-01` (`OD-037-GUARD=A`, `OD-W2-DOCS=A`).

## Test contract

```powershell
uv run pytest tests/unit/runtime/nexus/test_od_037_guard_human_decision_persistence.py tests/unit/rag/vectorstore/test_frz_typ04_vectorstore_reflection_gate.py tests/unit/rag/document_loaders/test_legacy_runtime_document_tenant_scope.py -q
uv run ruff check tests/unit/runtime/nexus/test_od_037_guard_human_decision_persistence.py tests/unit/rag/vectorstore/test_frz_typ04_vectorstore_reflection_gate.py tests/unit/rag/document_loaders/test_legacy_runtime_document_tenant_scope.py
uv run python -m scripts.governance.verify_plan_approval governance/tickets/T-IGX-090-C.md --plan governance/tickets/plans/PLAN_T-IGX-090-C.md
uv run python -m scripts.governance.verify_ticket_lifecycle governance/tickets/T-IGX-090-C.md --plan governance/tickets/plans/PLAN_T-IGX-090-C.md
uv run python -m scripts.governance.verify_ticket_approvals --ticket governance/tickets/T-IGX-090-C.md
```

## Out of scope

- Any edits under `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/`, or `docs/`.
- Any change to `persist_human_decision` (`OD-037-PERSIST` rejected under `OP-EC-01`).
- Any weakening of existing tests, `skip`, or `xfail`.
- Any declaration of `CLOSED`, `IIAC PASS`, `FRZ PASS`, or `G6 PASS`, or any `git push` / merge to `development`.
- C10/C15 environment changes, Docker migration, or vendor qualification changes.

## Lifecycle state

plan_approved
