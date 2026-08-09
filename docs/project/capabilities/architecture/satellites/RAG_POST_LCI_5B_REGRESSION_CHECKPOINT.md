# RAG post-LCI-5B regression checkpoint

Status: PASS

- Baseline checkpoint: `5379cb9ecd60de0d72744048bb8ea261334d5d09`
- Required ancestors: `48fbc2121c36d7545c02cbda04d7268827eedae2`, `5379cb9ecd60de0d72744048bb8ea261334d5d09`
- Fix commit SHA: `537ce6dd443529b4c837b79c403e078fae371f7a`
- R1 workspace propagation: `PRODUCTION_REGRESSION`, fixed in the document handler and top-level hierarchical bootstrap tenant forwarding.
- R2 native handles: `TEST_COMPATIBILITY_DEBT`; tests use `native_handle` and `attach_parser_native_handle`, with no `_docling_document` metadata tunnel.
- R3 lifecycle: `TEST_COMPATIBILITY_DEBT`; fixtures use native records and `add_records`; `add_documents` was not restored.
- R4 hierarchical tenant: `TEST_COMPATIBILITY_DEBT` for the old no-tenant fixture, with explicit tenant propagation fixed and no implicit default.
- Gate D: `NOT_APPLICABLE`; legacy full-answer stack remains removed, while retrieval and citation/source replacement boundaries passed.

## Proof and gates

- Scope propagation passed for `workspace-a` and `workspace_id=None`.
- Reserved parser metadata remained rejected.
- Native ingest → `VectorStoreRecord` → `InMemoryVectorStore` → scoped retrieval isolated same-ID content across workspaces.
- Add → delete → re-add produced no ghost result and exactly one vector.
- Regression tests: 106 passed.
- Gate rerun mapping A-Q: 122 passed.
- Gates: A PASS, B PASS, C PASS, D N/A, E PASS, F PASS, G PASS, H PASS, I PASS, J PASS, K PASS, L PASS, M PASS, N PASS, O PASS, P PASS, Q PASS.

## Golden, soak, and environment

- Golden retrieval: 6/6 PASS; raw minimum recall 0.0; all thresholds PASS.
- Soak: 4 workers × 2 queries, 8 queries; PASS, recall floor 1.0, no exceptions.
- `docling_core` was not installed and packaging was unchanged; native-handle tests used local-safe handles.
- `docx2txt`, `unstructured`, and `fastembed` optional paths were excluded.
- Windows SQLite/temp-artifact tests ran sequentially.
- Structural audits: inventory, LangChain boundary, KnowledgeDocument conformance, hierarchical bootstrap, and tenant-storage isolation PASS.

## Roadmap

- LCI-5A: APPROVED
- LCI-5B: APPROVED
- RAG-REGRESSION-GATE-1: PASS / CLOSED
- LCI-5C started: no
- Next: LCI-5C
