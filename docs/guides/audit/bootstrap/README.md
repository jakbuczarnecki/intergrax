# Bootstrap paste files

Copy **entire file** into a **new** Cursor agent chat as the first message.

| File | Mode | Scope |
|------|------|-------|
| [`01_audit_all_domains.txt`](01_audit_all_domains.txt) | A — audit-only | All 22 domain pairs |
| [`02_audit_one_domain.txt`](02_audit_one_domain.txt) | A1 — audit-only | Single domain (`DOMAIN=` line) |
| [`03_implement_plan_all_domains.txt`](03_implement_plan_all_domains.txt) | B — implement plan | All 22 |
| [`04_implement_plan_one_domain.txt`](04_implement_plan_one_domain.txt) | B1 — implement plan | Single domain |
| [`05_closeout_all_domains.txt`](05_closeout_all_domains.txt) | C — layer completion | All 22 (LCM 1–6) |
| [`06_interactive_layer_by_layer_audit.txt`](06_interactive_layer_by_layer_audit.txt) | **A2 — interactive audit** | One domain per stop; operator confirms before next |

**Mode A2 (interactive):** Same audit depth as Mode A, but **one domain per session stop**. Agent presents gaps, plan-vs-code findings, and proposed tasks, then **waits for operator confirmation** before the next domain. Use when pair-reviewing harness maturity layer-by-layer (ideal → canon → plan → code). Operator may approve plan rows and request commit between domains.

**Initialize run (recommended):**

```bash
uv run python scripts/init_architecture_audit_run.py --date YYYY-MM-DD --mode audit_only
```

**Resume:** paste same file + line `RESUME: docs/guides/audit/results/YYYY-MM-DD/progress.json`

**Mode B iteration:** one P0/P1 item per domain (or skip); agent continues through all 22 domains without asking between pairs unless operator writes `pause` / `stop`. A single Cursor turn may not finish the full run — use **Resume** for the next session. See [`IMPLEMENT_ORCHESTRATOR.md`](../IMPLEMENT_ORCHESTRATOR.md) §Iteration discipline.

**Canonical docs:** [`../README.md`](../README.md) · [`../ORCHESTRATOR.md`](../ORCHESTRATOR.md)
