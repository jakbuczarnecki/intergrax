# Cursor session bootstrap (paste files)

**Doc map:** [`../DOCUMENTATION_MAP.md`](../DOCUMENTATION_MAP.md) — bootstrap vs audit vs audit_results roles.  
**F3 session rule:** `ONE DOMAIN = ONE NEW CHAT` — see [`../guides/CURSOR_TOKEN_SETUP.md`](../guides/CURSOR_TOKEN_SETUP.md).  
**F2 setup:** root `AGENTS.md` is a Cursor auto-load stub; full reference in `docs/guides/AGENT_INSTRUCTIONS.md`. Keep `intergrax-iteration.mdc` always-on — do not delete stub via Settings trash icon.

Copy **entire file** into a **new** Cursor agent chat as the first message.

| File | Mode | Scope |
|------|------|-------|
| [`01_audit_all_domains.txt`](01_audit_all_domains.txt) | A — audit-only | All 22 domain pairs |
| [`02_audit_one_domain.txt`](02_audit_one_domain.txt) | A1 — audit-only | Single domain (`DOMAIN=` line) |
| [`03_implement_plan_all_domains.txt`](03_implement_plan_all_domains.txt) | B — implement plan | All 22 |
| [`04_implement_plan_one_domain.txt`](04_implement_plan_one_domain.txt) | B1 — implement plan | Single domain |
| [`05_closeout_all_domains.txt`](05_closeout_all_domains.txt) | C — layer completion | All 22 (LCM 1–6) |
| [`06_interactive_layer_by_layer_audit.txt`](06_interactive_layer_by_layer_audit.txt) | **A2 — interactive audit** | One domain per stop; operator confirms before next |
| [`idea_audit.txt`](idea_audit.txt) | **I — idea intake audit** | Single idea in chat — live audit; Cursor rule auto-triggers; on approval update architecture + plan |
| [`07_ci_preflight.txt`](07_ci_preflight.txt) | **CI — preflight before push/merge** | Run local parity with `.github/workflows/unit-tests.yml` (`--profile all`) |

**Mode I (idea audit):** Write the idea in natural language in a **new** chat (e.g. `Zrób audyt pomysłu: …`). Agent loads `.cursor/rules/intergrax-idea-audit.mdc` on trigger → procedure [`idea_audit.txt`](idea_audit.txt). **No** file editing, **no** `init_architecture_audit_run.py`, **no** `audit_results/`. See [`IDEA_AUDIT_ORCHESTRATOR.md`](../audit/IDEA_AUDIT_ORCHESTRATOR.md).

**Context budget (all audit modes):** Each `docs/audit/<DOMAIN>.md` includes **§0 Context budget** — scoped plan/architecture reads, one domain per chat recommended, listed gate scripts only. Never load full multi-thousand-line plan files.

**Mode A2 (interactive):** Same audit depth as Mode A, but **one domain per session stop**. Agent presents gaps, plan-vs-code findings, and proposed tasks, then **waits for operator confirmation** before the next domain.

**Initialize run (Modes A / B / C only — not Mode I):**

```bash
uv run python scripts/init_architecture_audit_run.py --date YYYY-MM-DD --mode audit_only
```

**Resume:** paste same file + line `RESUME: docs/audit_results/YYYY-MM-DD/progress.json`

**Mode B iteration:** one P0/P1 item per domain (or skip); agent continues through all 22 domains without asking between pairs unless operator writes `pause` / `stop`. See [`IMPLEMENT_ORCHESTRATOR.md`](../audit/IMPLEMENT_ORCHESTRATOR.md) §Iteration discipline.

**Canonical procedure docs:** [`../audit/README.md`](../audit/README.md) · [`../audit/ORCHESTRATOR.md`](../audit/ORCHESTRATOR.md)
