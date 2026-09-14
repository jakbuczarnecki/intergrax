# EE-B3-C — Execution Security Adversarial & Abuse Certification

**Task:** EE-B3-C  
**Branch:** `development`

## Provenance

| Field | Value |
|-------|-------|
| **START_HEAD** | `65a71baa96e4dba6a9ea74a347dc50ba8f9a7008` |
| **START_ORIGIN** | `65a71baa96e4dba6a9ea74a347dc50ba8f9a7008` |
| **TESTED_REMOTE_SHA** | `65a71baa96e4dba6a9ea74a347dc50ba8f9a7008` |

## Deliverables

| Artifact | Path |
| -------- | ---- |
| Adversarial abuse model | `docs/project/maintainers/architecture/EXECUTION_ENGINE_SECURITY_ADVERSARIAL_ABUSE_MODEL.md` |
| Abuse-case tests | `tests/unit/runtime/architecture/test_ee_b3_c_*.py` |
| Test helpers | `testing_support/security/` (test-only) |

**PRODUCTION CODE CHANGED:** NO (expected).

## Abuse scenarios (summary)

| Case | Typed rejection / control | Side-effect counter |
| ---- | ------------------------- | ------------------- |
| AC-01 | `ValueError` / format validators | N/A |
| AC-02 | `REJECT_TENANT` / `REJECT_IDENTITY` | N/A |
| AC-03 | `DelegationAuthorityError` | N/A |
| AC-04 | `PolicyAction.DENY` | N/A |
| AC-05 | `MeaningfulSideEffectAuthorizationRequiredError` / tool fail | executor = 0 |
| AC-06 | `ValueError` bundle consistency | N/A |
| AC-07 | lifecycle unchanged cross-tenant | N/A |
| AC-08 | `REJECT_AUTHORITY` / `REJECT_TENANT` | N/A |
| AC-09 | `StaleCheckpointWriteError` / `REJECT_IDENTITY` | N/A |
| AC-10 | `DeclarativePolicyHitlRequiredError` | executor = 0 |
| AC-11 | `ToolGovernanceDeniedError` | executor = 0 |
| AC-12 | `InvalidFanOutError` | orchestration not started |

## Findings

| Severity | Count |
| -------- | ----- |
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 0 |
| LOW | 0 |

## NPSC-5F SECURITY HANDOFF

**NONE** — protected surfaces not modified in EE-B3-C.

## Regression matrix

EE-A1, EE-A2 H1–H3, NPSC-4.2, NPSC-5B, NPSC-5E security slice, EE-B1.1–B1.3, EE-B2, EE-B3-A, EE-B3-C — see session pytest log.

## Static quality

`ruff check`, `ruff format --check`, `pyright` on changed scope — see final report.

## Final verdict

**PASS** _(pending pytest + static gates in session closeout)_.
