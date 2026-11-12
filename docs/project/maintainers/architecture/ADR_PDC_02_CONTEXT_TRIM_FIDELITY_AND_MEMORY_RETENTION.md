# ADR-PDC-02 — Context trim fidelity, durable compaction closure and memory retention enforcement

| Field | Value |
| --- | --- |
| **Document ID** | ADR-PDC-02 |
| **Status** | **Proposed** (pending OD-PDC-ADR-01) |
| **Campaign** | PDC-X P3 (T-IGX-036-D) |
| **Domain** | PDC-D2 Context & Knowledge |
| **Requirements** | PDC-D2-R011, PDC-D2-R012, PDC-D2-R015 (and PDC-D6-R022 for memory) — see [`PDC_DOMAIN_REQUIREMENTS_MATRIX.md`](../qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md) |
| **FRZ** | FRZ-TRC-05, FRZ-REC-04, FRZ-STA-06, FRZ-SEC-09 |

## Context

- `TOKENIZER_HARD_TRIM` in `runtime/nexus/context/degradation_ladder.py` rebuilds `ChatMessage` with role/content/metadata only, dropping `entry_id`, `name`, `tool_calls`, `tool_call_id`.
- CE-02 (compaction + retention) has a qualification suite (`tests/qualification/ce_02`) but no closure record; the gap audit lists it PARTIAL.
- `MemoryRetentionPolicy` is decision-only (no scheduler, no TTL); MEMORY-DURABILITY-LIFECYCLE is ACCEPTED/PLANNED.
- The CE-02 ownership table (`CONTEXT_ENGINEERING.md` §147–165) stays authoritative.

## Decision (proposed)

- Hard trim preserves message identity and tool-call linkage: truncate content only, keep `entry_id`, `name`, `tool_calls`, `tool_call_id`; if a tool-call pair cannot fit, drop the pair atomically (never orphan a call or result).
- PDC closes CE-02 by requalifying `tests/qualification/ce_02` at the PDC audit SHA plus adversarial durable-compaction round-trip tests; no new context engine.
- Memory retention enforcement runs through `MemoryControlPlane` (`forget` / `delete` paths already exist) driven by a typed retention sweep that also tombstones vector-index entries; the sweep is a background task admitted through canonical execution (no new scheduler).

## Alternatives rejected

- A second context assembly or compaction engine.
- Retention enforced inside individual memory providers (bypasses the single owner).

## Verification

`tests/qualification/pdc_x/d2/`: hard-trim linkage preservation, compaction round-trip + reconstruction, retention expiry with vector tombstone, mandatory instruction retention under pressure.

## Status note

Proposed only. No implementation before OD-PDC-ADR-01 accepts this ADR; product changes then follow OD-PDC-SEQ-01 (merge after MCP-SERVER-X P8).
