# Legal agent — architecture (scaffold baseline)

**Status:** Hard reset complete (Phase AA-LEG). Legacy pipeline removed; UAEP scaffold is the only implementation path.

## Purpose

Tier-2 contract review capability (`legal.review`) composed into `legal_application` via Nexus / UAEP.

## Layout

| Path | Role |
|------|------|
| `legal_agent.py` | `Agent` + UAEP steps |
| `contract.py` | `AgentContract` + `LEGAL_CONTRACT_REVIEW` skill |
| `steps/pipeline.py` | Domain stub pipeline |
| `SPEC_FROM_LEGACY.md` | Requirements extracted from pre-reset code |

## Configuration

- **Capabilities:** `legal.review`
- **Skills:** `legal.contract_review` (see `intergrax/skills/providers/legal/`)
- **Tools:** Resolved by Tier-3 `ApplicationEnvironmentProfile` — not imported in this package

## Runtime

1. Tier-3 host builds registry via `wire_application_environment`.
2. Nexus selects agent by capability.
3. UAEP executes `legal_step` → `run_domain_step` → Nexus pipeline stub.

## Next implementation steps

Port behaviors from `SPEC_FROM_LEGACY.md` as additional `@step` functions (one PR per step).
