# T-AF2-RW-TRUST — train parity disposition (Package C)

**Train compared:** `origin/arch-freeze/rw-train` @ `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a`  
**Ticket:** `governance/tickets/T-IGX-051-C.md` / `PLAN_T-IGX-051-C.md` §7.1 (`NT-TRUST-01`..`06`)

## Required behavior vs train

| NT id | Plan path / test | Train @ `e9bebe0ac` |
| --- | --- | --- |
| NT-TRUST-01 | `tests/unit/context/test_trust_fail_closed.py::test_nt_trust_01_rag_fragment_uses_context_prefix` | present; pytest **pass** |
| NT-TRUST-02 | `tests/unit/llm/test_messages.py::test_structured_transport_detection` | present; **pass** |
| NT-TRUST-03 | `tests/unit/context/test_context_formatter.py::test_formatter_emits_source_tagged_blocks` | present; **pass** |
| NT-TRUST-04 | `test_trust_fail_closed.py::test_nt_trust_04_whitespace_tool_call_id_rejected` | present; **pass** |
| NT-TRUST-05 | `test_trust_fail_closed.py::test_nt_trust_05_planner_ledger_includes_tool_payload_tokens` | present; **`intergrax/context/planner.py`** @ `1c2ebea6e` |
| NT-TRUST-06 | `tests/unit/tools/providers/rag/test_rag_scope.py::test_resolve_tenant_scoped_vectorstore_fails_closed_without_tenant` | present @ `66a3046ce`; **pass** |

Product paths cited in plan §3 indicative table (`intergrax/llm/messages.py`, `intergrax/context/planner.py`, context egress): last train-touch for planner trust slice `1c2ebea6e`; messages touched in trust-adjacent history `a915448ff`.

Note: plan shorthand `graph_assembly.py` maps to **`intergrax/runtime/nexus/context/graph_assembly.py`** on the train (not under `intergrax/context/`); NT-TRUST-01..05 contract tests above are green without an additional RW-TRUST product delta on `e9bebe0ac`.

## Disposition

**Skip implementation:** the ticket’s §7.1 negative-test contract is already satisfied on the train at `e9bebe0ac`. No further product commit under `T-IGX-051-C` in Package C.