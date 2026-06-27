---
id: IJ-2026-06-10-023
date: 2026-06-10
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-PROD-6
  - ACP-PROD-7
  - ACP-PROD-8
  - ACP-PROD-9
  - ACP-PROD-10
  - ACP-PROD-11
status: completed
commit: pending
adr: none — closes §40.6–§40.11 production gate scaffolding
---

# Artifact refs, threat CI, release gates, and schema registry

## Operator request

Execute the next ACP sprint: Wave 7 production gates (ACP-PROD-6 through ACP-PROD-11).

## Summary

Added typed `ArtifactRef` on `AgentRunResult`, PII redaction on policy verdict reasons, threat-model CI script, release gate aggregator, ACP CI conformance matrix checker, and contract schema version registry with validation script.

## Project impact

Production promotion path now has typed artifacts, privacy redaction baseline, and aggregate CI gates aligned with architecture §40.6–§40.11.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.6–§40.11 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 7 steps 7.6–7.8 |

## Changed artifacts

- `intergrax/contracts/artifact_ref.py`, `privacy_redaction.py`, `migrations/registry.py`
- `intergrax/contracts/agent_run.py`, `intergrax/agents/authoring/acp_run.py`
- `scripts/maintenance/check_agent_threat_model.py`, `check_agent_release_gates.py`, `check_acp_ci_conformance_matrix.py`, `check_contract_schema_versions.py`
- `tests/unit/contracts/test_acp_prod_artifact_privacy.py`

## Verification

```bash
uv run python scripts/maintenance/check_contract_schema_versions.py
uv run python scripts/maintenance/check_agent_threat_model.py
uv run python scripts/gates/check_acp_ci_conformance_matrix.py --scripts-only
uv run pytest tests/unit/contracts/test_acp_prod_artifact_privacy.py -q -m gate
```

Result: pass.

## Risks and follow-ups

- Full `check_acp_ci_conformance_matrix.py` execution in CI may be heavy — workflow uses `--scripts-only` baseline.
- Golden/regression eval suites per agent remain host-specific beyond scoreboard.
- Compensation enqueue (ACP-PROD-3 follow-up) still deferred.
