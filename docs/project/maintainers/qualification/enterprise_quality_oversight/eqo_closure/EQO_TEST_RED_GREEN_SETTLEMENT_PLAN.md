# EQO — Known-red settlement (KR-01..KR-28)

**Parent:** `EQO-X-06` · **Source:** `arch_freeze_program/KNOWN_RED_REGISTER.md` @ `DOC_PIN` · **Baseline ref:** `7960ae7f` (register draft).

## Rules (execution)

1. Classify each KR into one of **11 operator reason classes** (aligned with W0-03 taxonomy where applicable).
2. Pin for run: `TRAIN_PIN` **or** #680 Actions log — state explicitly per row.
3. **FIX** and **UNKNOWN** remain `BLOCKED` for freeze until MEP owner shows invariant N exclusion.
4. This file is a **result register**, not approval to skip/xfail.

## Skeleton result table

| KR ID | Draft class | EQO class (11-way) | Pin / log ref | pytest nodes enumerated | Owner node | Settlement status |
|-------|-------------|-------------------|---------------|-------------------------|------------|-------------------|
| KR-01 | ENV | TBD | TBD | 19 VPI modules (collect) | UNASSIGNED | NOT_STARTED |
| KR-02 | ENV | TBD | TBD | 5 psycopg modules | UNASSIGNED | NOT_STARTED |
| KR-03 | FIX | TBD | TBD | `test_reference_harness_strict.py` | T-AF2-RW-GATES | NOT_STARTED |
| KR-04 | FIX | TBD | TBD | 7 governance-tier nodes | T-AF2-RW-GATES | NOT_STARTED |
| KR-05 | ENV | TBD | TBD | diagnostic script (no node) | UNASSIGNED | NOT_STARTED |
| KR-06 | UNKNOWN | TBD | TBD | diag-final otel | UNASSIGNED | NOT_STARTED |
| KR-07 | FIX | TBD | TBD | durable-diagnostics script | UNASSIGNED | NOT_STARTED |
| KR-08 | FIX | TBD | TBD | 15 governance-fast nodes | IIAC rebind | NOT_STARTED |
| KR-09 | FIX | TBD | TBD | 9 EBH-7/F10 (enumerate PT-Q11) | PT-Q11 | NOT_STARTED |
| KR-10 | FIX | TBD | TBD | partial RRX counter test | PT-Q04/Q05 | NOT_STARTED |
| KR-11 | FIX | TBD | TBD | TRACE-X P6 parity | PT-Q06 | NOT_STARTED |
| KR-12 | UNKNOWN | TBD | TBD | 3 dist_scale (ids TBD) | PT-Q09 | NOT_STARTED |
| KR-13 | FIX | TBD | TBD | 77 incident (ids TBD) | T-AF2-RW-GATES | NOT_STARTED |
| KR-14 | FIX | TBD | TBD | EX-L-13 scenario trace | T-AF2-RW-GATES | NOT_STARTED |
| KR-15 | FIX | TBD | TBD | 14 COMPAT-X order-dep | UNASSIGNED | NOT_STARTED |
| KR-16 | FIX | TBD | TBD | 3 stale RAG tests | UNASSIGNED | NOT_STARTED |
| KR-17 | FIX | TBD | TBD | vendor_sdk gate | UNASSIGNED | NOT_STARTED |
| KR-18 | FIX | TBD | TBD | contract_to_implementation | UNASSIGNED | NOT_STARTED |
| KR-19 | FIX | TBD | TBD | DEP1 + p6 xfail | UNASSIGNED | NOT_STARTED |
| KR-20 | UNKNOWN | TBD | TBD | trace_x gates (4+) | UNASSIGNED | NOT_STARTED |
| KR-21 | ENV | TBD | TBD | optional extras cases | none | NOT_STARTED |
| KR-22 | UNKNOWN | TBD | TBD | ent_audit launcher timeout | UNASSIGNED | NOT_STARTED |
| KR-23 | UNKNOWN | TBD | TBD | MCP R-P7-02 (11) | T-AF2-RED-CLASSES | NOT_STARTED |
| KR-24 | FIX | TBD | TBD | scaffold MCP xfail | UNASSIGNED | NOT_STARTED |
| KR-25 | ENV | TBD | TBD | 27 materialization (MAX_PATH) | none | NOT_STARTED |
| KR-26 | UNKNOWN | TBD | TBD | F-W3-001 B5 tree | T-AF2-RED-CLASSES | NOT_STARTED |
| KR-27 | UNKNOWN | TBD | TBD | embedding ollama (2) | UNASSIGNED | NOT_STARTED |
| KR-28 | FIX | TBD | TBD | 2 llm_routing acceptance | PT-G01 | NOT_STARTED |

## Acceptance (X-06)

- All 28 rows: `EQO class` filled and `Settlement status` ∈ {`RESOLVED`, `BLOCKED`, `ENV_PROVEN`, `WAIT_MEP_NODE`}.
- Link each `BLOCKED` row to W2 ticket or exact blocker ID.
- Update closure matrix **C14** at X-06 CLS.