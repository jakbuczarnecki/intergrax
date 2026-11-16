# ADR-MCP-07 — Stage qualification via MCP-ACC (default: no new FRZ family)

| Field | Value |
| --- | --- |
| **Document ID** | ADR-MCP-07 |
| **Status** | **Proposed** (pending OD-MCP-07) |
| **Related OD** | OD-MCP-07 |

## Context

PLATFORM FRZ checklist remains **225** criteria. ENT-AUDIT-X and TOOL-LIFE-X historical MCP tests do not close MCP-SERVER-X. ECAP Mode A expects incremental evidence on existing families.

## Decision (proposed)

**Default path (OD-MCP-07-A):**

- Stage acceptance = **MCP-ACC-01..20** matrix green on `TEST_EXECUTION_SHA`.
- FRZ evidence = **incremental notes** on existing IDs (BND, OWN, CTR, EXE, SEC, TEN, CMP, REL, …) — **no new `FRZ-MCP-*` rows** and **no counter migration**.

**Alternate path (OD-MCP-07-B):** new FRZ family — **out of default campaign**; requires separate validator/ECAP/checklist migration ticket.

## Alternatives

| Option | Outcome |
| --- | --- |
| A. Incremental FRZ only | **Recommended** |
| B. New FRZ-MCP family | Operator-only; not in P1 default |

## Consequences

- P7 produces exact-SHA package; P8 IIAC still required for stage CLOSED.
- Implementer cannot self-certify audit or IIAC.
