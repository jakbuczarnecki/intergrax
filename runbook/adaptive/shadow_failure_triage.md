# Shadow failure triage

Operational runbook for investigating shadow-mode adaptive candidates that fail promotion or verification (Phase W-ADAPT-5.9).

## When to use

- Shadow allocation succeeded but utility does not beat baseline.
- Profile remains in `shadow` status beyond expected window.
- Canary promotion blocked by governance or golden scenario gate.

## Prerequisites

- Shadow runs tagged with `candidate_profile_version_id` in trace metadata.
- SignalStore contains SHADOW eval_mode signals for the task class.

## Triage checklist

1. **Signal coverage** — confirm ≥ 95% completed shadow runs emit `HarnessOutcomeSignal`.
2. **Utility comparison** — inspect mean U for SHADOW vs OFFLINE/ONLINE signals in `signal_trends.json`.
3. **Regression flags** — check for `step_explosion`, `llm_cost_spike`, or `tool_usage_drop`.
4. **Eval registry** — review release comparison delta in evaluation registry trends.
5. **Cost** — confirm normalized cost stays within budget envelope.
6. **Security** — re-run V-SEC adversarial baseline if policy or routing changed.

## Resolution paths

| Finding | Action |
|---------|--------|
| Insufficient shadow samples | Extend shadow window; do not promote |
| Utility below baseline | Retire draft; open new recommend cycle |
| Regression spike | Block loop kind; investigate ExecutionGuard history |
| Golden scenario fail | Fix evaluation assets before re-shadow |
| Cost overrun | Route to cost anomaly bridge recommendations |

## Escalation

- Harness architect for envelope policy changes.
- Security for policy-learning-related shadow failures.
- Agent author for task-class-specific quality regressions.

## Related artifacts

- `build/adaptive_harness/signal_trends.json`
- `build/adaptive_harness/verification_report.json`
- [ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md](../docs/ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) §12
