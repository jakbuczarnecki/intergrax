# PLAN T-IGX-051-C — T-AF2-RW-TRUST scaffold (ARCH-FREEZE development tor)

Status: **plan §7 filled (autonomy fala 3, 2026-11-21)**. Source: `MEP-AF2_v2.1.md` row T-AF2-RW-TRUST; **OD-AF-AUTO-03** product go after digest refresh.

## 1. Purpose

Remediate trust-boundary findings before RW / G2 on train; on `development`, establish ticket SSOT so parallel work in `af-dev-rw-trust` can proceed after plan approval.

## 2. Prerequisites

- `SCOPE` (train) / RC-1 admission on `development` documentation track.
- Operator decisions on trust-class model where PDC marks **Operator decision** (EX-D2-01/02/07).

## 3. Indicative write set

| Finding cluster | Primary paths |
|-----------------|---------------|
| EX-D2-01 | `intergrax/llm/messages.py`, `graph_assembly.py`, context fragment consumers |
| EX-D2-02 | `intergrax/context/` egress and sensitivity defaults |
| EX-D2-05 | `intergrax/context/planner.py` |
| EX-D2-07 | `intergrax/llm/` orphan `tool_call_id` handling |
| EX-L-05 / D2-R004 | RAG trust tests under `tests/unit/rag/` where root cause is product-bound |

## 4. Test contract (placeholder)

- Red-first: injection cannot place untrusted content in system role (fixture TBD).
- STRICT mode refuses orphan tool correlation ids (TBD node ids).
- Regression list to be copied from PDC / closed-world census in plan revision 2.

## 5. Out of scope

Same as ticket.

## 6. Next steps

Architect session fills node ids, regression command, and digest; operator approval; then implementation in `af-dev-rw-trust`.

## 7. Test contract (fala 3 — verified node ids)

PDC / MEP rows: EX-D2-01, EX-D2-02, EX-D2-05, EX-D2-07, EX-L-05, D2-R004. Cross-ref PLAN_T-IGX-043-C §5 naming (`NT-FC3-*` style). Logs: `.tmp/session/arch-freeze/REGRESSION-78-fala3/`.

### 7.1 Negative tests (`NT-TRUST-*`)

| Node id | PDC | Assertion | Pytest node id (verified collect) | Status |
|---------|-----|-----------|-------------------------------------|--------|
| NT-TRUST-01 | EX-D2-01 | RAG/workspace fragments never bare `role=system` without `[context:` prefix | `tests/unit/context/test_trust_fail_closed.py::test_nt_trust_01_rag_fragment_uses_context_prefix` | **green** (formatter contract) |
| NT-TRUST-02 | EX-D2-01 | plain `system` without `[context:` forces structured transport | `tests/unit/llm/test_messages.py::test_structured_transport_detection` | **green** (control) |
| NT-TRUST-03 | EX-D2-02 | Formatter tags fragment source in system block | `tests/unit/context/test_context_formatter.py::test_formatter_emits_source_tagged_blocks` | **green** |
| NT-TRUST-04 | EX-D2-07 | invalid `tool_call_id` on envelope build fails closed | `tests/unit/context/test_trust_fail_closed.py::test_nt_trust_04_whitespace_tool_call_id_rejected` | **green** (fala 4 slice 2) |
| NT-TRUST-05 | EX-D2-05 | planner tool-group ledger includes tool payload tokens | `tests/unit/context/test_trust_fail_closed.py::test_nt_trust_05_planner_ledger_includes_tool_payload_tokens` | **green** (fala 4 slice 2) |
| NT-TRUST-06 | EX-L-05 / D2-R004 | tenant-scoped vectorstore resolution | `tests/unit/tools/providers/rag/test_rag_scope.py::test_resolve_tenant_scoped_vectorstore_fails_closed_without_tenant` | **green** (fala 4 slice 3; `uv sync --extra dev-ci-rag`) |

Red-first package (new file, fala 3 slice 1): `tests/unit/context/test_trust_fail_closed.py` (`NT-TRUST-01`).

### 7.2 Regression command (trust + FC-3 bundle)

```text
uv run pytest tests/unit/context/test_trust_fail_closed.py tests/unit/context/test_context_formatter.py tests/unit/llm/test_messages.py -o cache_dir=<scratch>
```

| Slice | Scope | Status |
|-------|--------|--------|
| fala B2 slice 4 | §7.2 trust bundle only (excludes `NT-TRUST-06` / RAG collect path) | **green** — 64 passed @ `ec2957a56` (2026-11-21) |
| fala 5A slice 3+ | `NT-TRUST-06` + tenant-scope pytest (`dev-ci-rag`) | **green** — 1 + 11 passed @ `dc083f6ea` (2026-11-21) |

Plus FC-3 regression bundle from `PLAN_T-IGX-043-J` §7.2 (`regression_scope_fc3_bundle.txt`) before merge (serial `af-dev-merge`).

### 7.3 ENV

`uv sync --extra dev-ci --extra dev-ci-rag` for NT-TRUST-06 (`langchain-core`); pytest without `-p no:cacheprovider` (use `-o cache_dir=<scratch>` — repo temp plugin sets `cache_dir`).

### 7.4 Push

`uv run python -m scripts.governance.igx_gov ticket` with `Ref: governance/tickets/T-IGX-051-C.md`.