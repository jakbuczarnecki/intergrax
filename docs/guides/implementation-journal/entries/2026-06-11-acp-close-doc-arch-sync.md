---
id: IJ-2026-06-11-002
date: 2026-06-11
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-DOC-2
  - ACP-CLOSE-DOC-3
  - ACP-CLOSE-DOC-4
status: completed
commit: pending
adr: none — documentation sync only; no runtime contract change
---

# ACP-CLOSE sprint 1 — architecture and audit doc sync

## Operator request

Execute the first ACP-CLOSE sprint per plan §6.1bb: synchronize architecture canon with delivered ACP implementation and regenerate the domain audit prompt after the 2026-06-11 compliance audit.

## Summary

Updated `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §28.1–§28.3 (code map, maturity scorecard, GAP register with Closed/Open status), §21/§29/§30.8/§31.2 implementation maps, §36.4 alignment table, §37.8 maturity, §40 header and §40.13 audit acceptance. Marked plan rows ACP-CLOSE-DOC-2..4 **Done**. Refreshed `generate_domain_audit_prompts.py` AGENT overrides and regenerated `guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md`.

## Project impact

Architecture ↔ plan ↔ code traceability is restored after Phase ACP completion. Auditors and agents now see **32 Closed / 3 Open** GAP-ACP items and **ACP-CLOSE** as the active queue instead of stale „Planned/pre-implementation” labels.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §28.3 · §36.4 · §40.13 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-CLOSE-DOC-2..4 · §6.1bb |
| ADR | none |
| Audit / gap | GAP-ACP-03/04/07 remain Open |

## Changed artifacts

- `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — GAP register, code maps, maturity gates
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — DOC-2..4 Done; §6.1bb queue updated
- `scripts/generate_domain_audit_prompts.py` — AGENT active phases and known gaps
- `docs/guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` — regenerated

## Verification

```bash
uv run python scripts/check_docs_domain_pairs.py
uv run python scripts/generate_domain_audit_prompts.py
```

Result: pass (docs-only; `check_implementation_journal.py` passes after INDEX row added).

## Risks and follow-ups

- Next sprint: **ACP-CLOSE-LEG-1/2** (DEBT-ACP-06/04) or **ACP-CLOSE-PROD-1/2** (checkpoint on product hosts).
- Architecture file may need another pass when LEG/PROD code lands (no ADR required for doc-only sync).
