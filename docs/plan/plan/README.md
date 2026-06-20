# Platform Foundation — plan satellites

**Parent hub:** [`../PLATFORM_FOUNDATION.md`](../PLATFORM_FOUNDATION.md)

Load **one** satellite per session when a task or audit gap ID requires it. Do not bulk-read all files.

| File | When to load |
|------|----------------|
| [`PLATFORM_FOUNDATION_06_closed_queues.md`](PLATFORM_FOUNDATION_06_closed_queues.md) | Re-validating a **closed** §6.1/§6.2 queue cited in audit |
| [`PLATFORM_FOUNDATION_master_registers.md`](PLATFORM_FOUNDATION_master_registers.md) | Gap ID points to ORCH/FLOW/TS/… master register |
| [`PLATFORM_FOUNDATION_06_phase_detail.md`](PLATFORM_FOUNDATION_06_phase_detail.md) | Appendix L/M/N detail, §6.4 historical milestones |
| [`PLATFORM_FOUNDATION_phase_closeout.md`](PLATFORM_FOUNDATION_phase_closeout.md) | Phase V-REM, FAUDIT-32 closeout narrative |
| [`PLATFORM_FOUNDATION_appendices.md`](PLATFORM_FOUNDATION_appendices.md) | Appendices B–M traceability |

Regenerate hub split: `uv run python scripts/split_platform_foundation_plan.py` (after hub edits that add closed queues/registers).
