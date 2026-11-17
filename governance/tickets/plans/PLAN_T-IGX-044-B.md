# PLAN T-IGX-044-B — EQO-A remediation node PT-E01 (T-ENF-1): workflow prerequisites for the enforcement trial

Status: plan for T-IGX-044-B, drafted 2026-11-16 by an implementation session. The approval state is recorded in the ticket's Approved section and lifecycle state, not in this plan.

## 1. Purpose

Remove two workflow-level unknowns before any GitHub setting is changed (EQO-A enforcement plan, stage S1; nodes PT-E01 then PT-S01):

1. The two ENT-AUDIT-X workflows still run on the floating label `ubuntu-latest` and do not pin the Python that `uv` uses. Pin the runner to `ubuntu-24.04` and the uv Python to `3.12`, as T-IGX-040-C (`9fe9c9944`) did for the other uv-based workflows.
2. The job `governance-policy-pin` carries `continue-on-error: true`. Decision DP-65 (Operator, 2026-11-16): remove it if the script is stable and accept that promotion `development` to `main` stays blocked until the IIAC re-pin. Stability was checked from read-only GitHub data (section 4); the result is that the script is stable, so the line is removed.

This ticket certifies nothing, changes no GitHub setting, branch protection or ruleset, promotes no FRZ criterion and touches no product path.

## 2. Change set (3 workflow files, 13 lines, plus ticket and plan)

| File | Change |
|---|---|
| `.github/workflows/enterprise-audit.yml` | `ent-audit-fast` and `ent-audit-pr-gate`: `runs-on: ubuntu-latest` to `ubuntu-24.04`; `astral-sh/setup-uv@v5` gets `with: python-version: "3.12"` in both jobs |
| `.github/workflows/enterprise-audit-nightly.yml` | `ent-audit-nightly`: the same two changes |
| `.github/workflows/unit-tests.yml` | job `governance-policy-pin` only: delete the line `continue-on-error: true` (1 deletion). Allowed by the node scope (DP-65: the job is defined here); no other line of this file changes |

Design notes:

- `setup-uv@v5` is kept (no action-version bump). Its `python-version` input exists at `v5` and "sets UV_PYTHON" (`astral-sh/setup-uv@v5` `action.yml`, read with `gh api`). `requires-python = ">=3.12,<3.13"` (`pyproject.toml:14`) and `.python-version` = `3.12`.
- `UV_PYTHON_DOWNLOADS: never` is deliberately NOT added to the enterprise-audit workflows. They have never set it, and the nightly run `43848339391` shows `uv` downloading CPython 3.12.16 on `ubuntu-26.04` and succeeding. Adding it would change behaviour beyond the node scope.
- The step name of the pin step still reads "ADVISORY — job may fail without blocking merge policy" (`unit-tests.yml`, step `GOV-06 trusted policy pin compare`). It is a label, not a check context; it is left unchanged to keep the diff to the decision (see section 7).

## 3. Verification of the already-pinned workflows (not touched)

At `origin/development` `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2`, `9fe9c9944` is an ancestor (`git merge-base --is-ancestor`). `runs-on: ubuntu-24.04` is present in `llm-adapters-guard.yml:18`, `llm-network-smoke.yml:18`, `rag-guard.yml:25` and in every job of `unit-tests.yml` except `platform-plugin-windows-e2e` (`windows-latest`, line 240). Not touched by this ticket.

Two further workflows still use `ubuntu-latest` and are outside this node (`harness-release.yml:11`, `sync-repository-metadata.yml:23`; both set `python-version: "3.12"`; neither is a required check). Left as they are; listed as a follow-up question in the PR body.

## 4. Evidence (read-only; raw data under `.tmp/session/eqo-a-remediation/IMPL/PT-E01/`)

| Fact | Value | Source |
|---|---|---|
| Runner image of the nightly | `ubuntu-26.04` (so `ubuntu-latest` already resolves to 26.04); `uv` downloaded CPython 3.12.16; run succeeded | `gh run view 43848339391 --log` (2026-11-16 10:23 UTC), `nightly_43848339391.log` |
| Enterprise-audit PR runs | 11 of the last 12 succeeded, 1 cancelled | `gh run list --workflow enterprise-audit.yml` |
| Consequence | the pin is for determinism, not a repair of a failing run; the first CI run of this PR is the first run of these two workflows on `ubuntu-24.04` | this plan |
| `governance-policy-pin` history | 60 `unit-tests.yml` runs, 2026-11-15 17:13 to 2026-11-16 14:13 UTC: the step `GOV-06 trusted policy pin compare` ran 54 times (50 PR, 4 push), 54 successes, 0 script failures; 4 job failures were all the step `Install dependencies (CI minimal)` in the runner incident 2026-11-16 05:28-05:52 UTC (fixed by T-IGX-040-C); 1 skipped, 1 cancelled | `policy_pin_jobs.json` (`gh api actions/runs/<id>/jobs`) |
| Pin state at head | all 7 pinned files match `governance/TRUSTED_POLICY_PIN.yaml` (SHA-256 of the git blob at `2ab57e36d`) | `git show ... \| sha256sum` |
| Pin maintenance practice | `61ee57824` (2026-11-07 10:25:59) changed pinned files, `2f6664d53` refreshed the pin 41 s later; `bcbacf993` (2026-11-08) changed a pinned file and the pin together; no later commit changed a pinned file | `git log` |
| Where it is required | ruleset "Protect main" (20844442) lists `governance-policy-pin` among three required contexts; the `development` ruleset requires none of them | `gh api repos/.../rulesets/20844442` |
| Effect of `continue-on-error` on the check run | the check run `governance-policy-pin` has `conclusion: failure` on `5a05fc461` and `16febee4c` (install step failed) although the job carries `continue-on-error: true` | `gh api commits/<sha>/check-runs` |

The last row contradicts the enforcement plan's statement that the job "is a no-op" (WP-7 plan, section 0 item 5): job-level `continue-on-error` did not turn the check run green in the two cases that could be observed. What the flag changes, per GitHub semantics, is the workflow-run result, not the check-run result. This is recorded, not relied upon: the decision is taken on the stability data above.

Stability verdict: the script did not fail in any of the 54 executions and the pinned files match at head; DP-65's condition ("if stable") is met. The window is 21 hours (60 runs); `gh run list --limit 200` reaches back only to 2026-11-13, so a longer history was not examined.

## 5. Effect and risk

- Merge effect of this PR: none on `development` (no required check there). The change becomes visible as a red `governance-policy-pin` job and a red `Regression gate` run when a PR changes a pinned governance file without refreshing `TRUSTED_POLICY_PIN.yaml` (fix: `uv run python scripts/governance/verify_trusted_policy_pin.py --write-pin` in the same PR, as in `2f6664d53`).
- Promotion `development` to `main` stays blocked while `governance-fast` is red by design after the IIAC pin (WP-7 plan, section 0 item 4); that is the consequence the Operator accepted in DP-65. This ticket does not change `governance-fast`.
- `ent-audit-fast` and `ent-audit-pr-gate` run for the first time on `ubuntu-24.04` in this PR's CI. If either fails for an image reason, the PR must not be merged until that is understood; no result is claimed before CI has run.
- `.github/` is inside the EBH-7 executable-identity allowlist (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py:20-24`); the ticket and plan files under `governance/` are not, as for the other EQO-A tickets already on `development`. Whether an EBH-7 replay is affected was not run and is not established.
- No test or script changes. No secret, credential or vendor endpoint is involved.

## 6. Test contract

One process at a time, none that runs a workflow:

- Static YAML parse of the three workflows and a check that exactly the intended keys changed (`git diff --stat`, `git diff` review; stdlib-safe `python -I` run of `yaml.safe_load` where PyYAML is present). `actionlint` is not installed and was not used.
- `python -I scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-044-B.md` for the digest recorded in the ticket. `scripts/governance/verify_plan_approval.py` compares that digest first; the approval line itself is outside the implementer's remit and its presence is read from the ticket. `igx_gov ticket` (it wraps the fast regression set) is not run by this session.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three workflows, this ticket and this plan.
- CI on the PR itself is the only execution evidence for the pinned runner.

## 7. Open items and exclusions

- Independent audit: required, performed by node PT-AUD-ENF1 (independent G6 session, implementer excluded; plan WP-12 `DAG.md`). This ticket does not close any finding by itself.
- The stale step label "ADVISORY — job may fail without blocking merge policy" in `unit-tests.yml` is not changed here (labels are outside the DP-65 decision); the audit or a later ticket may rename it.
- Trial ruleset on `refs/heads/enforcement-trial/**` (AR-45, AR-50), Stage 1 required checks (AR-46) and any other GitHub setting: out of scope, separate authorisation after this PR is merged.
- `harness-release.yml` and `sync-repository-metadata.yml` still on `ubuntu-latest`: question for the Operator whether to pin them in a later ticket.

## 8. Rollback

Revert the single PR (workflow text only). No data, setting or code path is affected.

## 9. Closure condition

PR merged after review and after the CI of this PR is understood; independent audit by PT-AUD-ENF1; the ticket records the merge SHA. No finding is marked closed or verified by this ticket.
