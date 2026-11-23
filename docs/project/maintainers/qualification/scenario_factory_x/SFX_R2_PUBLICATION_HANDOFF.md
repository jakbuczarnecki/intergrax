# SFX R2 — Publication handoff (IGX-SFX-R2-PUBLISH-001)

**Status:** `NOT_PUBLISHED` · **Branch:** `integ/sfx-r2-corr-001`  
**Worktree:** `D:/Projekty/_wt/intergrax/sfx-r2-corr-001`

## Git pins

| Field | SHA / value |
| --- | --- |
| `CURRENT_REMOTE_BASE_SHA` | `3e5dedf267bc783a795ef20974a4e70809f05fbd` |
| `SOURCE_CORRECTIVE_STACK` (local `development`) | `85f9e955ad06aa0c91a342d62827483fce8caf3f` |
| `PUBLICATION_CANDIDATE_SHA` | `4b63b8dafb6ec9323d9c573666f21f9500ec8bea` |
| `REMOTE_PUBLICATION_SHA` | **NONE** |
| Replay commits on branch | `022dc251f` → `a89e2673d` → `4b63b8dafb6e` |

**Scope note:** source commit `e5cd9879e` included 3 out-of-scope paths (`enterprise_audit`, `test_xd_05_host_background`); publication replay **excludes** them (evidence: `.tmp/proof/sfx-r2-publish-001/git-scope-19files.txt`).

## Validation @ `4b63b8dafb6e` (agent session)

| Check | Result | Evidence |
| --- | --- | --- |
| JSON (6 schemas) | PASS | worktree parse + CF-4 integers |
| DAG (37 tasks) | PASS | `.tmp/session/sfx-r2-publish-001` WBS script |
| REQ-R2 (18 rows) | PASS | `SFX_REQUIREMENTS_TRACEABILITY_R2_PROPOSAL.json` |
| REF-1 oracles | PASS | `SFX_R2_REF_GEN_EVALUATION_PLAN.md` |
| Architecture | PASS | reviewer @ 4b63b8dafb6e |
| SFX local links | PASS | `.tmp/session/sfx-r2-publish-001/sfx_link_check.py` |
| Global link pytest | GLOBAL_PREEXISTING_FAILURE | non-SFX docs |
| Independent audit | PASS_WITH_FINDINGS | [18ab5a33](18ab5a33-3572-4963-b477-c6ac3f457dfc) |

## Governance (no fabricated approval)

| Ticket | Lifecycle | Plan | Publication of docs |
| --- | --- | --- | --- |
| `T-IGX-070-A` | `plan_pending` | `PLAN_T-IGX-070-A` — no `Approved:` | R2 package already on `origin` @ `10cf95df8`; corrective overlay proposes only |
| `T-IGX-070-B` | `plan_approved` | `PLAN_T-IGX-070-B` — no `Approved:` | **Proposed documentation** may be pushed as branch/PR; **not** canonical R2 adoption |

`verify_ticket_lifecycle.py`: **PASS** on both tickets @ main checkout.  
Formal `plan_approved` / `in_progress` promotion requires operator `Approved:` + plan digest.

## Operator procedure (order)

1. **Review** [`SFX_R2_CORRECTIVE_DOSSIER.md`](./SFX_R2_CORRECTIVE_DOSSIER.md) and [`r2/SFX_R2_INDEPENDENT_AUDIT_MATRIX.md`](./r2/SFX_R2_INDEPENDENT_AUDIT_MATRIX.md).
2. **(Optional, for FSM promotion)** Add `Approved:` on `PLAN_T-IGX-070-B` when adopting the corrective documentation batch — **not** required to read the branch, but required for ticket state beyond `plan_pending`.
3. **Publish branch** (from any checkout with network + hooks):

```powershell
git fetch origin development
git push -u origin docs/sfx-r2-corr-001:docs/sfx-r2-corr-001
```

4. **Open PR** `docs/sfx-r2-corr-001` → `development` (no force, no hook bypass).
5. **CI:** `ticket-governance`, regression `fast`, docs gates as configured on PR.
6. **After merge:** run `uv run python -m scripts.governance.igx_gov ticket` on clean `development`; pin `PUBLISHED_SHA` in dossier (separate commit — SHA ladder).
7. **Separate decisions (not in this PR):** `CANONICAL_R2_APPROVED`, SEF delegation OD, ECAP §16 ADR.

## Post-publication checklist

- [ ] Remote contains `4b63b8dafb6e` (or merge commit) on `development`
- [ ] `git-scope-19files.txt` matches `git diff --name-only <base>..<merge>`
- [ ] Update `PUBLISHED_SHA` / `REMOTE_PUBLICATION_SHA` in corrective dossier
- [ ] Re-run SFX package validation on merged SHA
- [ ] Keep `IMPLEMENTATION_AUTHORIZED = NO`

**Recommendation:** `PUBLISH_READY` (documentation branch; governance approval for ticket promotion remains operator-led).
## Reconciliation (2026-11-23)

- Governance bootstrap: delegation `7e999074c`, approval `3e5dedf26` on `origin/development`.
- `BASE_SHA` for integrator: `3e5dedf267bc783a795ef20974a4e70809f05fbd`.
- Ticket `Approved:` lines are on `development` only; SFX PR must not add/remove approval lines.
