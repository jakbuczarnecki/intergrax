# PLAN T-IGX-069-A — EQO W0-01 / W0-05 documentation publication (isolated)

**Ticket (proposed):** `governance/tickets/T-IGX-069-A.md` — scaffold on operator approval only.  
**Lifecycle:** draft — no `Approved:` line in this artifact.

## Objective

Publish doc-only W0 closure artifacts for **EQO-W0-01** (live pin bind) and **EQO-W0-05** (counter reconciliation embedded in bind metadata), without mixing train/product commits or PT-T01.

## Scope boundary

| In scope | Out of scope |
|----------|----------------|
| EQO-W0-01 pin refresh | EQO-W0-02..04 re-implementation |
| EQO-W0-05 publication evidence via pin bind | FRZ checklist edits |
| | Roadmap / WBS status coercion |
| | FCS / M3 promotion |
| | T-IGX-065-A commits |
| | Commits 2bc18ac1..b4146cbf (train/W3 delta) |

## Read set (verification only)

- `.tmp/session/enterprise-roadmap-stage02b/W0_01_PROPOSED_PIN_BIND.json`
- `.tmp/session/enterprise-roadmap-stage02b/W0_05_COUNTERS.json`
- `.tmp/session/enterprise-roadmap-stage02b/W0_05_ACCEPTANCE_CHECK.md`
- `.tmp/session/enterprise-roadmap-stage02b/W0_05_PUBLICATION_PACKET.md`
- `.tmp/session/enterprise-roadmap-stage02a/W0_01_PIN_RECONCILIATION.json`
- `.tmp/session/enterprise-roadmap-stage02a/W0_05_COUNTER_RECONCILIATION.json`
- `C:\Users\XPS\.cursor\plans\eqo_closure_master_7499b85e.plan.md` (§2 counters)
- `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/EQO_MASTER_COMPLETION_WBS.json`
- `git fetch origin --prune` + `git rev-parse origin/development` + `git rev-parse origin/arch-freeze/rw-train`

## Exact write set (closed)

| Path | Action | Justification |
|------|--------|---------------|
| `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/EQO_PIN_BIND.json` | replace content from `W0_01_PROPOSED_PIN_BIND.json` with `publication_sha` set to merge commit on `development` after authorized publish | mandatory W0-01; carries fetch date, live `actual` refs, ancestry, historical bind preservation, explicit `match: false` where pins diverge; W0-05 counter qualification referenced in session evidence (no separate eqo_closure counter file required by W0-05 packet) |

No other paths. No optional addenda unless operator issues a scope amendment ticket.

## Acceptance

### EQO-W0-01

- Historical `expected` DOC/TRAIN pins unchanged (`8cee3c5a…`, `28c27cb8…`).
- Live `actual` equals post-fetch `origin/development` and `origin/arch-freeze/rw-train`.
- `fetch_verified: true` and provenance string names authorized publish session.
- `historical_bind_2026_11_21` block preserved.
- No `match: true` when `expected != actual`.
- `FCS` and `M3` remain null; `PR680_MERGE.not_fcs` / `not_c16_pass` true.
- Ancestry checks pass (`git merge-base --is-ancestor` for historical pins and PR680 merge).

### EQO-W0-05

- Counter reconciliation in stage02a/02b matches master plan §2 (151 WBS tasks, A 240, B2 186, EDER 1073, FRZ 225/OPEN 109 scoped snapshot).
- Publication evidence: `publication_sha` on canonical bind + governance ticket Ref on commit.
- Doc-only diff vs `origin/development`; no product or test paths.

## Test contract

On isolated publish checkout at `origin/development` after applying write set only:

```text
uv run python -m scripts.governance.igx_gov ticket
```

Expected: ticket Ref present; regression tier `fast` green for doc-only change (operator environment with `uv`).

## Rollback

Revert the single merge/ commit touching only `EQO_PIN_BIND.json` on `development` (no force push).

## Publication requirements

1. Operator `Approved:` on this plan digest on ticket T-IGX-069-A.
2. Isolated environment: new worktree or PR branch from **exact** `origin/development` (`3334b34e` at session fetch); **exclude** all seven local commits ahead of origin.
3. Apply write set; verify `git diff --name-only` equals one file.
4. Independent review (non-implementer): write-set, pins, no mixed commits, no false CLOSED.
5. Publish via approved PR or ff push per repository policy (operator / integrator).

## Publication verification (post-merge)

- Record `publication_sha` inside `EQO_PIN_BIND.json`.
- Confirm remote `origin/development` ancestry includes merge.
- Re-run W0-05 acceptance table against live tree at `publication_sha`.

## Operator authorization line (placeholder)

```text
Approved: <operator signature> — digest <computed below>
```