# Rollback active adaptive profile

Operational runbook for restoring the previous harness profile version after a failed verification or manual ops decision (Phase W-ADAPT-5.9).

## When to use

- VerificationLoop reports failure on an active or canary profile.
- Golden scenario utility drops below baseline threshold.
- ExecutionGuard regression rate spikes during verify window.
- Manual ops decision after incident review.

## Prerequisites

- Access to adaptive harness stores under `build/adaptive_harness/`.
- `ProfileActivePointerStore` contains both `active_version_id` and `previous_version_id`.
- Runbook owner: harness platform / SRE on-call.

## Steps

1. Confirm failure in `build/adaptive_harness/verification_report.json`.
2. Identify tenant, task class, and artifact type from the failed `VerificationResult`.
3. Invoke rollback via `AdaptationExecutor.rollback()` (lab) or ops API when exposed:

```python
executor.rollback(
    tenant_id="tenant-a",
    task_class="golden-echo",
    artifact_type=ProfileArtifactType.RAG,
)
```

4. Verify pointer swap: active version equals previous baseline version.
5. Confirm `ADAPTIVE_PROFILE_ROLLBACK` runtime event in trace export.
6. If auto-apply was blocked, review `loop_apply_blocks` store before re-enabling the loop kind.

## Success criteria

- Active profile pointer restored within **5 minutes** (KPI: mean rollback time).
- No further auto-apply attempts for blocked loop kinds until ops clears the block.
- Post-rollback utility returns to baseline within the next verify window.

## Related artifacts

- `build/adaptive_harness/verification_report.json`
- `build/adaptive_harness/l4_runtime_evidence.json`
- [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](../docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) §9.6
