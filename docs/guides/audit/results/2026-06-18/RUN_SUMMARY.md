# Interactive layer-by-layer audit run — 2026-06-18

**Mode:** audit_only (Mode A2 batch) · **Scope:** layers 6–22 (continued from layers 1–5 prior session)

## Status

**Complete** — all 22 domain pairs audited; §6.1av maintenance queues registered per domain.

## Rollup (layers 6–22 this batch)

| # | Domain | Verdict | MAINT IDs | Commit (batch) |
|---|--------|---------|-----------|----------------|
| 6 | AGENT_CONTRACTS_AND_ASSEMBLY | L3+ | ACP-MAINT-01..03 | ae86fb40 |
| 7 | LLM_ADAPTERS | L3 | LLM-MAINT-01..04 | 726750cb |
| 8 | TOOLS | L3 | TOOL-MAINT-01..04 | a5f5fbb7 |
| 9 | CODE_CRAFT | L3 | ECC-MAINT-01..04 | 0a0c22f1 |
| 10 | SKILLS | L3 | SK-MAINT-01..04 | a7d20e41 |
| 11 | INTEGRATIONS | L3 | INT-MAINT-01..04 | e54d9936 |
| 12 | RAG | L3 | RAG-MAINT-01..04 | c270e7c1 |
| 13 | MEMORY | L3 | MEM-MAINT-01..04 | 6be6fdac |
| 14 | CONTEXT_ENGINEERING | L3+ | CE-MAINT-01..04 | d553fbb9 |
| 15 | MODALITY | L3 | MOD-MAINT-01..04 | 0b4c7543 |
| 16 | OBSERVABILITY | L3 | OBS-MAINT-01..04 | 370e21d0 |
| 17 | RELIABILITY_FAILURE_AND_HITL | L3 | REL-MAINT-01..04 | a845011b |
| 18 | CRITIC_VERIFICATION | L3 | CVL-MAINT-01..04 | 96107ac3 |
| 19 | ADAPTIVE_HARNESS_INTELLIGENCE | L3+ | AHI-MAINT-01..04 | (this batch) |
| 20 | ELASTIC_CAPACITY_AND_SCALING | L3 | ECP-MAINT-01..04 | (this batch) |
| 21 | EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE | L3 | DX-MAINT-01..04 | (this batch) |
| 22 | TIER3_APPLICATION_ENVIRONMENT | L3 | T3-MAINT-01..04 | (this batch) |

## Known test hygiene (tracked, not blocking L3 verdict)

- `MOD-MAINT-01/02` — 2 failing `tests/unit/model_inference/` tests (fix required)
- `RAG-MAINT-*` — Windows pytest teardown crash on full rag suite (environment)
- `boundary_demo` — AS-3 violation (ACP-MAINT-01)

## Policy

No P0/P1 opened. All MAINT rows **Planned** — implementation deferred to gate maintenance PRs. Phase K / §6.3 product work not started.
