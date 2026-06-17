# Bootstrap paste files

Copy **entire file** into a **new** Cursor agent chat as the first message.

| File | Mode | Scope |
|------|------|-------|
| [`01_audit_all_domains.txt`](01_audit_all_domains.txt) | A — audit-only | All 22 domain pairs |
| [`02_audit_one_domain.txt`](02_audit_one_domain.txt) | A1 — audit-only | Single domain (`DOMAIN=` line) |
| [`03_implement_plan_all_domains.txt`](03_implement_plan_all_domains.txt) | B — implement plan | All 22 |
| [`04_implement_plan_one_domain.txt`](04_implement_plan_one_domain.txt) | B1 — implement plan | Single domain |
| [`05_closeout_all_domains.txt`](05_closeout_all_domains.txt) | C — layer completion | All 22 (LCM 1–6) |

**Initialize run (recommended):**

```bash
uv run python scripts/init_architecture_audit_run.py --date YYYY-MM-DD --mode audit_only
```

**Resume:** paste same file + line `RESUME: docs/guides/audit/results/YYYY-MM-DD/progress.json`

**Canonical docs:** [`../README.md`](../README.md) · [`../ORCHESTRATOR.md`](../ORCHESTRATOR.md)
