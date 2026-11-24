# ARCH-FREEZE R0 reentry — verified root-cause record (2026-11)

**Date:** 2026-11-24  
**Packet:** P-R0-1W  
**Branch at verification:** `development`  
**`development` HEAD:** `b2f79d3bba6db8ea8d497d03b90792ea5d262ceb`  
**Verification:** read-only git replay; facts traced to session artefacts under `.tmp/session/arch-freeze/R0/`.

This document records verified findings for ARCH-FREEZE R0 reentry planning. It does not assert operator approval, certification, `ARCH_FREEZE_CLOSED`, or IIAC formal re-activation.

## S1 — IIAC not in force on `development` (ACT-N11)

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Activation record `status: IN_FORCE`, `audit_execution_sha` `6d88e7e381a9254ba2290d228ff741745fff3fec`, `corrective_implementation_sha` `9d3a042fcb7219ab344e526e036ae3e3ca3dde9e` | CONFIRMED | [`docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json`](../IIAC_POLICY_ACTIVATION_RECORD.json) |
| Supplemental rebind chain `352b1d901..82bbaacc8` | UNVERIFIED as the binding chain in the record | Record `effective_from_sha` is `44b4fb41f51e534b27e911d0279b4642b9245c18` |
| Diff `6d88e7e38..development` touches 10 product paths | CONFIRMED | [Ten product paths](#ten-product-paths) below |
| Gate raises ACT-N11 when product paths drift after audited SHA | CONFIRMED | `scripts/governance/iiac_activation_gate.py` L731–L740 — `product_paths_changed_between(audit_sha, HEAD)` |

### ACT-N11 gate citation

```731:740:scripts/governance/iiac_activation_gate.py
    if audit_sha and head:
        for rel in _material_changes(repo, audit_sha, head):
            errors.append(f"ACT-N11 stale audit after material governance change: {rel}")
        try:
            product_changes = product_paths_changed_between(audit_sha, head, repo=repo)
        except RuntimeError as exc:
            errors.append(str(exc))
            product_changes = []
        for rel in product_changes:
            errors.append(f"ACT-N11 product path changed after the audited SHA: {rel}")
```

## S2 — `development` hold breach

Hold text is published in [`MEP-AF2_v2.1.md`](./MEP-AF2_v2.1.md) L566–L575. Hold-start pin `a73d15ab3` is confirmed as the MEP E-pin (L112).

| Metric | Value |
| --- | --- |
| Product-path commits on `development` since hold pin `a73d15ab3` | **164** |
| Canonical commit listing | `.tmp/session/arch-freeze/R0/product_commits_since_hold.txt` (session artefact; not tracked in git) |
| Product-path commits since audit SHA `6d88e7e381a9254ba2290d228ff741745fff3fec` | **4** |
| Listing since `6d88e7e38` | `.tmp/session/arch-freeze/R0/product_commits_since_6d88e7e38.txt` |

`development` HEAD `b2f79d3bba6db8ea8d497d03b90792ea5d262ceb` is newer than the latest product-path commit in this range (`2c6d7af1a0ab1c05efeb99adbde2e47166439cf4`) and does not itself add a further product-path commit in the audited span.

### Four product-path commits since `6d88e7e38`

| SHA | Subject | Product paths touched |
| --- | --- | --- |
| `2c6d7af1a0ab1c05efeb99adbde2e47166439cf4` | ARCH-FREEZE: enforce fail-closed PT-WF tenant authority | `intergrax/debug/router.py`, `intergrax/experiments/workflow.py` |
| `a4f2b2677b88cfe8afef9ae409bae1c91a0ad8bd` | W2-RECOVERY-01: publish RW-PLAN, phase-check-3, PT-V04 and Campaign B tooling. | `applications/local_workspace_application/pyproject.toml` |
| `3f225b62699ac3897dec6e019cd5daad072c00eb` | Bump pytest from 8.4.2 to 9.0.3 | `applications/local_workspace_application/pyproject.toml` |
| `8409782c97ee23fda29fb4d727e14f51c5df04a8` | Close C1-EXIT residuals: tenant propagation through reference projection wiring, fleet readiness census, and LKW strict-product test fixtures. | `agents/boundary_demo/boundary_demo_agent.py`, `applications/local_workspace_application/host/reference_lifecycle_input.py`, `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py`, `applications/local_workspace_application/tests/host/test_local_workspace_environment_profile.py`, `applications/local_workspace_application/tests/hosting/test_hosted_foreground.py`, `applications/local_workspace_application/tests/lkw_ac3_projection.py`, `intergrax/applications/_shared/registry_projection_input_bundle.py` |

## S3 — autonomy decisions outside ODR

Records **OD-AF-AUTO-01** through **OD-AF-AUTO-07** appear in [`continuation/OPERATOR_DECISIONS_AUTONOMY.md`](./continuation/OPERATOR_DECISIONS_AUTONOMY.md) and are **not** present in [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](../ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md). This is a registry gap observation only; disposition remains with Architecture Authority and operator process outside this record.

## Ten product paths

Product-path files differing between `6d88e7e381a9254ba2290d228ff741745fff3fec` and `development` HEAD (`git diff --name-only 6d88e7e38 development`, product-path filter):

1. `agents/boundary_demo/boundary_demo_agent.py`
2. `applications/local_workspace_application/host/reference_lifecycle_input.py`
3. `applications/local_workspace_application/pyproject.toml`
4. `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py`
5. `applications/local_workspace_application/tests/host/test_local_workspace_environment_profile.py`
6. `applications/local_workspace_application/tests/hosting/test_hosted_foreground.py`
7. `applications/local_workspace_application/tests/lkw_ac3_projection.py`
8. `intergrax/applications/_shared/registry_projection_input_bundle.py`
9. `intergrax/debug/router.py`
10. `intergrax/experiments/workflow.py`

## Session source

Draft consolidated in `.tmp/session/arch-freeze/R0/reentry_draft.md` (packet P-R0-1, read-only verification session).