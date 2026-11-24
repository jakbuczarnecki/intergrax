# T-AF2-RW-PLAN-AUDIT — evidence index (AR-12-07)

| Artifact | Role |
| --- | --- |
| `docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json` | T-AF2-RW-PLAN machine output |
| `docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md` | Human render |
| `docs/project/maintainers/qualification/arch_freeze_program/known_red_register.v1.json` | Known-red FIX planning mirror |
| `docs/project/maintainers/qualification/arch_freeze_program/REV_MASTER_FORMAL_CLOSEOUT_RECORD.md` | REV-MASTER predecessor |
| `docs/project/maintainers/qualification/arch_freeze_program/continuation/CAMPAIGN_B_ALPHA_SCOPE_DEFECTS.md` | PT-B01/B02 SCOPE input |
| `scripts/governance/verify_rw_plan_phase_check3.py` | Mechanical phase check 3 |

**SHA:** `ebe45c26930a9291a3df2d6a75028c35c036cee2` (auditor session 2026-11-24). **No implementer verdict attached.**

## PT-Q02 train CI factual replay (2026-11-24)

| Artifact | Role |
| --- | --- |
| `.tmp/session/arch-freeze/pt-q02_gh_pr_checks_811.log` | `gh pr checks 811` @ merge **876af7e5d** |
| `.tmp/session/arch-freeze/pt-q02_gh_pr_checks_812.log` | `gh pr checks 812` @ train tip **e9bebe0ac** |
| `.tmp/session/arch-freeze/pt-q02_pin_verify_2026-11-24.log` | `git rev-parse origin/development` + `origin/arch-freeze/rw-train` |
| [`TRAIN_CI_REPLAY_PLAYBOOK.md`](TRAIN_CI_REPLAY_PLAYBOOK.md) Runda 3 | Cited job URLs for `ci-smoke` / `rag-gate` / `governance-fast` |

**Not** `RED_FIRST_EVIDENCE` and not green `governance-fast`.

## RED_FIRST_EVIDENCE (still INCOMPLETE)

Mechanical rule (`scripts/governance/verify_rw_plan_phase_check3.py`, `validate_red_first_evidence`): every `code_fix_inventory` row in `rw_plan.v1.json` with `red_first_status: EVIDENCE_MISSING` needs a tracked `RED_AT_PREFIX` replay with a 40-character `red_first_prefix_sha` and a non-empty `red_first_evidence_ref` before `RED_FIRST_EVIDENCE` can complete. **39** / **39** `code_fix_inventory` rows remain `EVIDENCE_MISSING` @ `rw_plan.v1.json` on `development` @ `cdf0feece0c8e42a259f69784f0781b431b5937b` (2026-11-24 red-first replay wave; **0** prefixes verified red; no row greened). Replay used detached worktree `D:/Projekty/_wt/intergrax/af-red-first`, one `uv run pytest` per row, logs under `.tmp/session/arch-freeze/red-first/` (`REPLAY_RESULTS.json`, per-`frz_id` `.log`). Candidate prefix = parent of `git log -1 --diff-filter=A` on the named regression path; if the test file is absent at that SHA, the row stays `EVIDENCE_MISSING` (example: **FRZ-BND-04** / **FRZ-BND-05** — `pytest_exit_code=-2`, `test file absent at prefix` `754ee000675d71c1734656702beb0aef8ba58fd8`; **FRZ-BND-06** — same at `307a871c059f150a6b744e39135a737399467e88`). **SCOPE not closed.**

