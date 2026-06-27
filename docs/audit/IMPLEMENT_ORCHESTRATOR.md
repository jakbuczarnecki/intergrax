# Architecture Audit — Implement Plan Backlog (all domains)

**Status:** Canonical orchestrator for **Mode B** — execute open plan items, no fresh audit  
**Bootstrap:** [`bootstrap/03_implement_plan_all_domains.txt`](../bootstrap/03_implement_plan_all_domains.txt)  
**Single domain:** [`bootstrap/04_implement_plan_one_domain.txt`](../bootstrap/04_implement_plan_one_domain.txt)

---

## Purpose

Iterate all 22 domain pairs and **implement accepted backlog items** already registered in `docs/plan/<DOMAIN>.md`.

- Does **not** start with a full layer audit.
- Does **not** expand scope to Phase K / §6.3 without operator reprioritization.
- **One Cursor session bootstrap** → agent iterates domain-by-domain until all pairs are handled or operator writes `pause` / `stop`.
- **One iteration unit** = one domain: read plan → pick next open P0/P1 (or skip) → implement → verify → journal → next domain.

---

## Open item selection (P0 / P1)

A backlog row is **in scope** for Mode B when **both**:

1. **Status** column is not **Done** (e.g. Planned, Open, In progress, Backlog).
2. **Priority** column on that row is **P0** or **P1**.

**Out of scope** unless the operator names a specific row id:

- Rows with Priority **Medium**, **Low**, **Critical** (without P0/P1), or empty.
- Child rows under a wave header marked `(P1)` — the wave label does **not** inherit to children; use each row's Priority column.
- Phase K / §6.3 product backlog without explicit reprioritization.

When multiple P0/P1 rows are open in one domain, pick **one** coherent PR-sized slice (prefer P0 over P1). Remaining rows stay for a future run — do **not** implement a second item in the same domain in one orchestrated pass.

---

## Per-domain iteration

For current `<DOMAIN>`:

1. Read **only** `docs/plan/<DOMAIN>.md` + `docs/architecture/<DOMAIN>.md` (for contracts).
2. **If no open P0/P1 rows:** set `progress.json` → `status: skipped`, `verdict: no_open_p0_p1`, update `p0_open` / `p1_open` counts; brief checkpoint; **continue** to next domain.
3. **Else:** select **one** open P0/P1 row (prefer P0, then P1).
4. Implement minimal diff; reuse Tier-0 mechanisms.
5. Run domain-relevant tests and gate scripts from plan row / `AGENTS.md`.
6. Update plan row status to **Done**.
7. Add implementation journal entry when behavior changed ([`implementation-journal/`](../implementation-journal/README.md)).
8. Update `results/<run_id>/progress.json` → `status: completed`, `item_id`, `plan_updated: true`.
9. Brief checkpoint; **continue** to next domain.

**Safety fuse:** plan requires architecture change not reflected in canon → **STOP**, report operator, set domain `blocked`.

---

## Iteration discipline

| Rule | Meaning |
|------|---------|
| **Atomic unit** | Finish the **entire** current domain (implement **or** skip) before starting the next |
| **One item per domain** | At most one P0/P1 implementation per domain per orchestrated pass |
| **No micro-pauses** | Do not ask the operator every 2 steps inside a domain |
| **Between domains** | Brief checkpoint; **continue without asking** unless operator wrote `pause` / `stop` |
| **No fork questions** | Do not ask "which domain next" or "stay vs continue" — follow `domain_order` and `current_domain` |
| **Commits** | Only when operator explicitly requests |
| **Tier boundaries** | [`AGENTS.md`](../../AGENTS.md) — never violate |

---

## Resume across Cursor sessions

A single agent turn may not cover all 22 domains. That is expected — use resume, not restart.

1. Paste the **same** bootstrap file (`03_implement_plan_all_domains.txt`).
2. Add one line: `RESUME: docs/audit_results/YYYY-MM-DD/progress.json`
3. Agent reads `current_domain` and continues — **do not restart from pair #1**.

---

## Progress file

Same layout as audit runs (`results/YYYY-MM-DD/progress.json`) with `"mode": "implement_plan"`.

Initialize:

```bash
uv run python scripts/audit/init_architecture_audit_run.py --date YYYY-MM-DD --mode implement_plan
```

Domain `status` values: `pending` · `completed` (item implemented) · `skipped` (no open P0/P1) · `blocked` (safety fuse).

---

## Run completion

When every domain in `domain_order` is `completed`, `skipped`, or `blocked`:

1. Write rollup in `results/<run_id>/RUN_SUMMARY.md`.
2. Set `completed_at` and `current_domain: null` in `progress.json`.
3. Run `uv run python scripts/audit/check_architecture_audit_run.py <run_id> --require-complete`.

---

## Commits

Only when operator explicitly requests.

---

## ADR

Follow domain plan row / [`AGENTS.md`](../../AGENTS.md) ADR rules per implementation item. No ADR for this orchestrator unless process semantics change platform contracts.
