# Full Harness Layer Completion Orchestrator

**Status:** Canonical orchestrator for **Mode C** — audit + closeout (LCM Steps 1–6)  
**Bootstrap:** [`bootstrap/05_closeout_all_domains.txt`](../bootstrap/05_closeout_all_domains.txt)  
**Per-domain LCM canon:** [`LAYER_COMPLETION_MODE.md`](../guides/LAYER_COMPLETION_MODE.md)

---

## Purpose

Sequentially run **Layer Completion Mode** on all 22 domain pairs until each reaches **Architecturally Mature** or **Frozen** (no open P0/P1 in domain scope).

This is the English canonical equivalent of the operator workflow previously kept under `docs/_external/`.

---

## Domain order

Same 22-pair order as [`ORCHESTRATOR.md`](ORCHESTRATOR.md).

---

## Progress file

Use `docs/audit_results/YYYY-MM-DD/progress.json` with `"mode": "layer_completion"`.

Extended fields per domain (optional): `recommendation`, `scores`, `journal`, `backlog_p2_p4` — see completed Full Harness LC runs.

Initialize:

```bash
uv run python scripts/init_architecture_audit_run.py --date YYYY-MM-DD --mode layer_completion
```

---

## Per-domain loop

For each `<DOMAIN>`, execute [`LAYER_COMPLETION_MODE.md`](../guides/LAYER_COMPLETION_MODE.md) Steps 1 → 1A → 2 → 3 → 4 → 5 → 6.

### Orchestration rules

| Rule | Content |
|------|---------|
| Docs scope | Only `architecture/<DOMAIN>.md` + `plan/<DOMAIN>.md` per iteration |
| P0/P1 | Block next domain until closed or operator reprioritizes |
| Mature/frozen | Short re-audit (Steps 1+6); skip full sprints if clean |
| Journal | English entry under `implementation-journal/entries/YYYY-MM-DD/` |
| Commits | Operator request only |

---

## Resume

Paste `05_closeout_all_domains.txt` + `RESUME: docs/audit_results/YYYY-MM-DD/progress.json`.

---

## Run completion

All domains `mature` or `frozen` → rollup `RUN_SUMMARY.md` → `completed_at` set.

Do **not** declare the entire platform permanently complete — only this closeout run.
