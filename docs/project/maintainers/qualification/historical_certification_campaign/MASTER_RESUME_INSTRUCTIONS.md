# MASTER resume instructions (refreshed 2026-11-08T11:15:30Z)
State: the campaign result is PARTIAL / EXACT BLOCKERS. The three PRs (405, 406, 407) are merged; the final package is published under `docs/project/maintainers/qualification/historical_certification_campaign/` (this directory). The local files `campaign_registry.json`, `build_final.py`, `build_report.py`, `dashboard.py`, `HOUSEKEEPING.md` and `worker-settings.json` are NOT part of the repository; MASTER keeps them under `.tmp/session/hist-cert-campaign/`.

1. Re-verify: `git fetch origin`; compare `origin/development` with `41908e3fb`; read new commits (the ENT-AUDIT-X stream is a separate session: never alter its material).
2. Source of truth for a new orchestrator: `campaign_registry.json` (local, gitignored) and `HISTORICAL_CERTIFICATION_MASTER_CLOSURE_MATRIX.json` (published). Regenerate this package with `python -I build_final.py` after updating the registry (counters derive from records).
3. Sessions: all campaign sessions finished. If one must be restarted: `hc run --bg --name <name> --cwd <worktree, forward slashes> --outlive-parent --no-notify -- claude --settings <campaign>/worker-settings.json --add-dir <campaign> --permission-mode auto`, wait until `hc get <name>` is idle, then `hc send` the brief. The `herdr` CLI server is NOT what herdr-app shows; do not use it.
4. Pending Operator decisions: see `HISTORICAL_AUTHORITY_AND_OPERATOR_DECISIONS.md` (DR-1, 2, 3, 4, 6, 8, 9 unanswered; DR-5, 7, 10, 11, 12 answered).
5. Held work: tickets `T-IGX-030-B` and `T-IGX-030-C` (plan_pending, no Approved line) wait for the IIAC re-audit / re-activation batch (ACT-N11). Do not approve or implement them before that.
6. Worktrees (do not delete without checking): hist01..hist04, impl-tlx, aud-030a, pin-ba07eba (read-only pinned checkout), int-docs, int-rules, int-final, mst-head-64ce093.
7. Housekeeping: HK-01 hung process tree (`HOUSEKEEPING.md`).
8. Language rule: all documentation, code and rules are English only; chat with the Operator follows the session language.
9. Publication rule: the Operator merges; verify `headRefOid` equals the pushed tip and read CI on that exact head before saying a PR is ready; push a new branch only after `git branch --set-upstream-to=origin/development` with an explicit refspec.
