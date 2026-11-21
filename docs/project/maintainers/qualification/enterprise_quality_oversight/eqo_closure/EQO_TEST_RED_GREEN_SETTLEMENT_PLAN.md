# EQO — Known-red settlement (KR-01..KR-28)

**Parent:** `EQO-X-06` · **Source:** `arch_freeze_program/KNOWN_RED_REGISTER.md` @ `TRAIN_PIN` `28c27cb` · **Baseline ref:** `7960ae7f` (register draft). Machine rows: `kr_settlement.jsonl`.

## Rules (execution)

1. Classify each KR into one of **11 operator reason classes** (aligned with W0-03 taxonomy where applicable).
2. Pin for run: `TRAIN_PIN` **or** #680 Actions log — state explicitly per row.
3. **FIX** and **UNKNOWN** remain `BLOCKED` for freeze until MEP owner shows invariant N exclusion.
4. This file is a **result register**, not approval to skip/xfail.

## Skeleton result table

| KR ID | Draft class | EQO class (11-way) | Pin / log ref | pytest nodes enumerated | Owner node | Settlement status |
|-------|-------------|-------------------|---------------|-------------------------|------------|-------------------|
| KR-01 | ENV | brak_SDK | TRAIN_PIN | 19 VPI modules (collect) | UNASSIGNED | BLOCKED |
| KR-02 | ENV | brak_SDK | TRAIN_PIN | 5 psycopg modules | UNASSIGNED | BLOCKED |
| KR-03 | FIX | defekt | 7960ae7f | `test_reference_harness_strict.py` | T-AF2-RW-GATES | BLOCKED |
| KR-04 | FIX | fixture | 7960ae7f | 7 governance-tier nodes | T-AF2-RW-GATES | BLOCKED |
| KR-05 | ENV | srodowisko | 7960ae7f | diagnostic script (no node) | UNASSIGNED | BLOCKED |
| KR-06 | UNKNOWN | nieustalony | 7960ae7f | diag-final otel | UNASSIGNED | BLOCKED |
| KR-07 | FIX | defekt | 7960ae7f | durable-diagnostics script | UNASSIGNED | BLOCKED |
| KR-08 | FIX | stale_gate | TRAIN_PIN / #680 | 15 governance-fast nodes | IIAC rebind | BLOCKED |
| KR-09 | FIX | stale_gate | 7960ae7f | 9 EBH-7/F10 (enumerate PT-Q11) | PT-Q11 | WAIT_MEP_NODE |
| KR-10 | FIX | stale_gate | 7960ae7f | partial RRX counter test | PT-Q04/Q05 | WAIT_MEP_NODE |
| KR-11 | FIX | defekt | 7960ae7f | TRACE-X P6 parity | PT-Q06 | WAIT_MEP_NODE |
| KR-12 | UNKNOWN | live_only | 7960ae7f | 3 dist_scale (ids TBD) | PT-Q09 | BLOCKED |
| KR-13 | FIX | regresja | 7960ae7f | 77 incident (ids TBD) | T-AF2-RW-GATES | BLOCKED |
| KR-14 | FIX | defekt | 7960ae7f | EX-L-13 scenario trace | T-AF2-RW-GATES | BLOCKED |
| KR-15 | FIX | kolejnosc | 7960ae7f | 14 COMPAT-X order-dep | UNASSIGNED | BLOCKED |
| KR-16 | FIX | stale_gate | 7960ae7f | 3 stale RAG tests | UNASSIGNED | BLOCKED |
| KR-17 | FIX | defekt | 7960ae7f | vendor_sdk gate | UNASSIGNED | BLOCKED |
| KR-18 | FIX | defekt | 7960ae7f | contract_to_implementation | UNASSIGNED | BLOCKED |
| KR-19 | FIX | defekt | 7960ae7f | DEP1 + p6 xfail | UNASSIGNED | BLOCKED |
| KR-20 | UNKNOWN | nieustalony | 7960ae7f | trace_x gates (4+) | UNASSIGNED | BLOCKED |
| KR-21 | ENV | brak_SDK | 7960ae7f | optional extras cases | none | BLOCKED |
| KR-22 | UNKNOWN | nieustalony | 7960ae7f | ent_audit launcher timeout | UNASSIGNED | BLOCKED |
| KR-23 | UNKNOWN | nieustalony | 7960ae7f | MCP R-P7-02 (11) | T-AF2-RED-CLASSES | BLOCKED |
| KR-24 | FIX | defekt | 7960ae7f | scaffold MCP xfail | UNASSIGNED | BLOCKED |
| KR-25 | ENV | srodowisko | 7960ae7f | 27 materialization (MAX_PATH) | none | BLOCKED |
| KR-26 | UNKNOWN | nieustalony | 7960ae7f | F-W3-001 B5 tree | T-AF2-RED-CLASSES | BLOCKED |
| KR-27 | UNKNOWN | nieustalony | 7960ae7f | embedding ollama (2) | UNASSIGNED | BLOCKED |
| KR-28 | FIX | stale_gate | TRAIN_PIN | 2 llm_routing acceptance | PT-G01 | BLOCKED |

## Acceptance (X-06)

- All 28 rows: `EQO class` filled and `Settlement status` ∈ {`RESOLVED`, `BLOCKED`, `ENV_PROVEN`, `WAIT_MEP_NODE`}.
- Link each `BLOCKED` row to W2 ticket or exact blocker ID.
- Update closure matrix **C14** at X-06 CLS.