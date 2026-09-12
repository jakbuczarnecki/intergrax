# Self-Healing R6.4 — Autonomy Qualification and Safety Validation

## Purpose

R6.4 adds a **read-only qualification layer** that verifies enterprise autonomy configuration **before** hosts rely on R6.1–R6.3 at runtime.

Qualification **evaluates posture**; it does **not**:

- activate autonomy,
- change policies,
- invoke executors,
- repair configuration,
- expand autonomy levels or authority.

## Placement

```text
Host configuration snapshot
        |
        v
AutonomyQualificationService (R6.4)
        |
        +-- ExecutionBoundarySafetyCheck
        +-- DefaultPolicySafetyCheck
        +-- AuditCapabilityCheck
        |
        v
AutonomyQualificationResult (+ optional AutonomyQualificationRepository)
```

Upstream R6 flow (unchanged):

```text
Recommendation → Autonomy Control → Decision Evaluation → Execution Guard → Execution Spine
```

## Contracts

| Artifact | Role |
|----------|------|
| `AutonomySafetyValidator` | Qualify configuration without side effects |
| `AutonomySafetyCheck` | Plugin for one safety aspect |
| `AutonomyQualificationResult` | Immutable outcome (`PASS` / `WARNING` / `FAILED`) |
| `AutonomyQualificationRepository` | Persistence port for audit replay (no vendor adapter in R6.4) |

## PASS criteria

Aggregate **PASS** when every registered check returns **PASS**:

- `AutonomyExecutionGuard` and `AutonomyExecutionBoundary` are present and protocol-satisfying.
- No `direct_executor_access_enabled` bypass flag.
- Default autonomy level is one of: `OBSERVE_ONLY`, `RECOMMEND_ONLY`, `APPROVAL_REQUIRED`.
- `AutonomyEvaluationAuditRecorder` and `AutonomyExecutionAuditRepository` ports are configured.

## FAIL criteria

Aggregate **FAILED** when any check reports **FAILED**, including:

| Code | Meaning |
|------|---------|
| `missing_execution_guard` | No guard before spine admission |
| `missing_execution_boundary` | No boundary integration port |
| `direct_executor_access` | Declared bypass of guard/boundary |
| `forbidden_full_autonomy_default` | `FULL_AUTONOMY` as default |
| `unsafe_controlled_execution_default` | `CONTROLLED_EXECUTION` as platform default |
| `missing_evaluation_audit_recorder` | R6.2 audit port absent |
| `missing_execution_audit_repository` | R6.3 guard audit port absent |

**WARNING** is reserved for checks that report non-blocking enterprise findings (none in the default bundle).

## Autonomy boundaries

- Qualification inspects **ports and declared defaults** only.
- Guard `check()` / boundary `authorize()` are **not** invoked during qualification.
- Execution authority remains on the existing **Execution Spine**; R6.4 does not add admission hooks or executors.

## Enterprise checklist

- [ ] Default policy level is advisory (`RECOMMEND_ONLY` or stricter).
- [ ] Guard and boundary wired before enabling controlled execution paths.
- [ ] Evaluation and execution audit ports reachable for replay.
- [ ] Qualification run recorded (`validation_id`, timestamp, checks, failures).
- [ ] No direct executor routing flag in host configuration.
- [ ] `FULL_AUTONOMY` remains contract-only (never default).

## Audit replay fields

`AutonomyQualificationResult` carries:

- `validation_id`
- `audit.qualified_at`
- `audit.checks_executed`
- per-check `AutonomySafetyCheckResult` and aggregated `AutonomySafetyIssue` list

Hosts persist via `AutonomyQualificationRepository` adapters.

## Extension

Additional checks (e.g. `PolicyConfigurationSafetyCheck`, `LegacyCompatibilityCheck`) implement `AutonomySafetyCheck` and are injected into `AutonomyQualificationService` — no second governance framework.
