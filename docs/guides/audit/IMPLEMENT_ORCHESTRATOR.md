# Architecture Audit — Implement Plan Backlog (all domains)

**Status:** Canonical orchestrator for **Mode B** — execute open plan items, no fresh audit  
**Bootstrap:** [`bootstrap/03_implement_plan_all_domains.txt`](bootstrap/03_implement_plan_all_domains.txt)  
**Single domain:** [`bootstrap/04_implement_plan_one_domain.txt`](bootstrap/04_implement_plan_one_domain.txt)

---

## Purpose

Iterate all 22 domain pairs and **implement accepted backlog items** already registered in `docs/plan/<DOMAIN>.md`.

- Does **not** start with a full layer audit.
- Does **not** expand scope to Phase K / §6.3 without operator reprioritization.
- **One iteration unit** = one domain: read plan → pick next open P0/P1 (or operator-selected row) → implement → verify → journal.

---

## Per-domain iteration

1. Read **only** `docs/plan/<DOMAIN>.md` + `docs/architecture/<DOMAIN>.md` (for contracts).
2. Select **one coherent** open item (prefer P0, then P1; one PR-sized slice).
3. Implement minimal diff; reuse Tier-0 mechanisms.
4. Run domain-relevant tests and gate scripts from plan row / `AGENTS.md`.
5. Update plan row status to **Done**.
6. Add implementation journal entry when behavior changed ([`implementation-journal/`](../implementation-journal/README.md)).
7. Update `results/<run_id>/progress.json`.
8. Continue to next domain unless `pause` / `stop`.

**Safety fuse:** plan requires architecture change not reflected in canon → **STOP**, propose doc update first.

---

## Progress file

Same layout as audit runs (`results/YYYY-MM-DD/progress.json`) with `"mode": "implement_plan"`.

Initialize:

```bash
uv run python scripts/init_architecture_audit_run.py --date YYYY-MM-DD --mode implement_plan
```

---

## Commits

Only when operator explicitly requests.
