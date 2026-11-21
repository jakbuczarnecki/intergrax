# PLAN T-IGX-043-J — PT-WF scaffold (ARCH-FREEZE development tor)

Status: **draft scaffold** on `development` (2026-11-21). Not approved for product implementation. Canonical wave description: `docs/project/maintainers/qualification/arch_freeze_program/WP12_TICKETS_DRAFT.md` row PT-WF; authorization **AR-44** in `WP12_AUTHORIZATION_REQUESTS.md`.

## 1. Purpose

Close PT-WF prerequisites before RW milestone: G-09 tenant-scoped evaluation registry on STRICT/PRODUCT harness hosts, experiment workflow governance identity (see `PLAN_T-IGX-041-E` finding F-D5-07 as related prior art), and conditional E2–E4 repairs per Campaign B-1.

## 2. Prerequisites

- `SCOPE` formally valid on train (blocked until `T-AF2-RW-PLAN-AUDIT` PASS).
- **PT-B01** diagnostics when required by DAG.
- No concurrent edit of `harness_host_runtime.py` with TENANT batch.

## 3. Write set (initial, to be detailed before `plan_pending`)

| Area | Paths (indicative) |
|------|---------------------|
| G-09 / harness | `applications/_shared/harness_host_runtime.py`, `evaluation_wiring.py` |
| E2 | `runtime/execution/runtime.py` |
| E3 | `compat/langchain/documents.py` |
| E4 | `experiments/store.py`, `debug/router.py`, `experiments/workflow.py` |

## 4. Test contract (placeholder)

- Red-first tests to be named in plan revision 2 (NT-WF-*).
- Regression: subset of ENT-A and evaluation qualification nodes (TBD from WP-4 `rows.jsonl`).
- `uv run python -m scripts.governance.igx_gov ticket` before push.

## 5. Out of scope

Same as ticket `T-IGX-043-J.md`.

## 6. Next steps (operator)

1. Expand sections 3–4 with file-level diffs and node ids (delegate or architect session).
2. Record plan digest in the ticket; move lifecycle to `plan_pending`.
3. Operator or delegate approval line; then `plan_approved` and product work in `af-dev-pt-wf`.

## 7. Revision 2 queue (autonomy fala 2 — architect minimum)

| Slice | Node / finding | Status | Blocker |
|-------|----------------|--------|---------|
| G-09 | EX-L-09 / FRZ-TEN-01 | **MERGED** @ `aac4b083b` (`evaluation_assembly_resolver`) | — |
| E2 | Campaign B-1 runtime path | **BLOCKED** | B-1 confirmation + file diffs |
| E3 | LangChain document compat | **BLOCKED** | node ids TBD |
| E4 | experiments workflow | **BLOCKED** | node ids TBD |
| Owner repairs | FRZ rows WP-4 §C (OWN/CTR/PLG/…) | **BLOCKED** | regression node census from `rows.jsonl` |

No further product commits until this section lists named `NT-WF-*` tests and a pytest command with verified node ids.