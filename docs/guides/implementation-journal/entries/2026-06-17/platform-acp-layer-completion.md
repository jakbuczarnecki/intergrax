---
id: IJ-2026-06-17-026
date: 2026-06-17
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-LC-S1
  - ACP-LC-S2
  - ACP-LC-S3
  - ACP-LC-S4
  - Full-Harness-LC-ACP
status: completed
commit: pending
adr: none — documentation and process closeout; ACP runtime already Done
---

# AGENT_CONTRACTS_AND_ASSEMBLY — Full Harness Layer Completion closeout

## Operator request

Commit REASONING_AND_COGNITION Full Harness LC work and continue orchestration to AGENT_CONTRACTS_AND_ASSEMBLY.

## Summary

- Confirmed ACP + ACP-CLOSE + ACP-FINISH + AUDIT-IDEAL **Done** — §28.3 **37 Closed · 0 Open**.
- Synced audit prompt known gaps (AUDIT-IDEAL-19.1/20.1/31.1 → Done).
- Updated ACP-INV-02 canon after ACP-CLOSE-LEG-5.
- Verified `check_agent_acp_close_ci.py` green (fleet 17/17, migration complete).

## Project impact

ACP domain formally closed for Full Harness LC orchestration — no open P0/P1; production gates and token budget depth already shipped.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §21, §28.3, §40.13 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` Phase ACP-LC |
| ADR | ADR-AGENT-001/002/003 (existing) |

## Changed artifacts

- `docs/guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` — known gaps sync
- `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — ACP-INV-02, §28.3 audit sync
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — Phase ACP-LC register

## Verification

```bash
python scripts/check_agent_acp_close_ci.py
```

## Risks and follow-ups

- `boundary_demo` remains legacy UAEP authoring (partner PoC) — P2 fleet semantic drift.
- COST-1 graph-level RunBudget cap — P2 cross-domain.
- FAUDIT-REG.1 eval registry depth — P2 PLATFORM_FOUNDATION owner.
