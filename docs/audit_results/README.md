# Architecture audit results

Dated outputs from **orchestrated** harness audits ([`ORCHESTRATOR.md`](../audit/ORCHESTRATOR.md)).

---

## Layout

```text
docs/audit_results/
  YYYY-MM-DD/
    progress.json       # machine state — resume across sessions
    RUN_SUMMARY.md      # human rollup
    <DOMAIN>.md         # per-domain result (required when status=completed)
    <DOMAIN>.json       # optional machine-readable findings
```

**Second run same day:** `YYYY-MM-DD_run-2/` (manual folder name).

---

## Initialize a run

```bash
uv run python scripts/init_architecture_audit_run.py --date 2026-06-17 --mode audit_only
uv run python scripts/init_architecture_audit_run.py --date 2026-06-17 --mode audit_only --domain MEMORY
uv run python scripts/init_architecture_audit_run.py --date 2026-06-17 --mode implement_plan
uv run python scripts/init_architecture_audit_run.py --date 2026-06-17 --mode layer_completion
```

---

## Validate

```bash
uv run python scripts/check_architecture_audit_run.py 2026-06-17
uv run python scripts/check_architecture_audit_run.py 2026-06-17 --require-complete
```

---

## Git policy

**Commit** `progress.json`, `RUN_SUMMARY.md`, and completed `<DOMAIN>.md` files — they are the audit history.

Do **not** commit chat transcripts. Operator PL notes may stay in `docs/_external/` (gitignored).

---

## Template

Copy [`TEMPLATE_DOMAIN_RESULT.md`](../audit/TEMPLATE_DOMAIN_RESULT.md) when writing `<DOMAIN>.md`.
