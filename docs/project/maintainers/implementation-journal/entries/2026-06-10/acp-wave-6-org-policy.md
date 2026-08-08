---
id: IJ-2026-06-10-020
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-ORG-1
  - ACP-ORG-2
  - ACP-ORG-3
  - ACP-ORG-4
  - ACP-ORG-5
status: completed
commit: pending
adr: none — implements existing architecture §39 envelope
---

# Organizational policy envelope and kernel enforcement

## Operator request

Execute the next ACP sprint: Wave 6 organizational policy (ACP-ORG-1 through ACP-ORG-5).

## Summary

Added `OrganizationalPolicyEnvelope` on `ApplicationEnvironmentProfile`, runtime `OrganizationalPolicyContext` via `merge_environment`, kernel org pre-checks for denied channels/tools, `ComplianceSummary` on `AgentRunResult`, and lab virtual-workforce profile preset with gate tests.

## Project impact

Tier-3 hosts can declare org-wide rules without agent forks; harness enforces channel/tool overlays and exposes measurable compliance rollups on typed ACP runs.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §39 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 6 steps 6.3–6.5 |
| Audit map | Layer 9 — Agent contracts / policy |

## Changed artifacts

- `intergrax/applications/contracts/org_policy.py` — envelope + lab fixture
- `intergrax/agents/org_policy_merge.py`, `intergrax/agents/compliance_summary.py`
- `intergrax/agents/run_environment.py`, `intergrax/agents/authoring/acp_run.py`
- `intergrax/runtime/policy/org_enforcement.py`, `intergrax/runtime/kernel/step_kernel.py`
- `intergrax/contracts/agent_run.py` — `ComplianceSummary`
- `tests/unit/agents/test_org_policy_merge_acp_org.py`, `tests/unit/runtime/kernel/test_step_kernel.py`

## Verification

```bash
uv run pytest tests/unit/agents/test_org_policy_merge_acp_org.py tests/unit/runtime/kernel/test_step_kernel.py -q -m gate
```

Result: pass.

## Risks and follow-ups

- Wave 7 production reliability (ACP-PROD-1..3) blocks mutating prod agents.
- Full `RuntimePolicyBundle` slice on `OrganizationalPolicyContext` deferred to UAEP policy depth.
- Golden eval suite per `compliance_profile_id` can expand beyond kernel happy-path tests.
