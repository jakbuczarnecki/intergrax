<!-- Campaign: EQO-A | BASELINE_SHA: 258bbf6abb6fdbb92ef0185d058527fbb1e47f04 | generated_utc: 2026-11-15T15:53:58Z | source_registry_sha256: 96fced0e0ddd428bcbf7897900558bbf4b3dcac3709c6dbc9fac207978b4e166 -->
# Campaign A (EQO-A) charter

Enterprise Quality Oversight, Campaign A: a read-only, exact-SHA baseline audit of the Intergrax Enterprise program. EQO-A is **not** a roadmap stage. It issues no certificate, promotes no FRZ criterion, closes no stage or ticket and changes no canonical status. The dossier is accepted (or not) only by a separate independent audit session.

## Operator decisions (binding, 2026-11-15)
- **OD-EQOA-01 Baseline.** `BASELINE_SHA` = `origin/development` at W0 start = `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, pinned in a detached worktree. The planning pin was `f87df3b2a93bd09aa8e1577b80122391249772c8`; the 116 later commits are listed in `BASELINE.json` and are all pre-baseline. Commits after the baseline go only into the post-baseline delta annex.
- **OD-EQOA-02 Publication.** Work under `.tmp/` during execution. After the independent auditor's verdict, one docs-only PR publishes to `docs/project/maintainers/qualification/enterprise_quality_oversight/campaign_a/`. The Operator merges it.
- **OD-EQOA-03 Tests.** At most 10 bounded, read-only pytest invocations in total (explicit node IDs or files, one process at a time, logs under `.tmp/`).

## Scope
In scope: all roadmap stage IDs (section 3 table, section 5 ledger, prose-only IDs); all FRZ-* IDs at the baseline; the certification and qualification records they cite; code and tests as far as needed to map concerns and check claims; governance tooling; all debt/exception/decision registers; PDC-X, ARCH-FREEZE, SFX, SEF-X, SCENARIO-READINESS-X and SCENARIO-GATE records for status and dependencies only. Exclusions are in `SCOPE_EXCLUSIONS.md`.

## Roles
- Executor: Sonnet session, Delivery DevOS role `orchestrator`. Runs W0-W5, prepares the W6 handoff, applies corrective loops, opens the publication PR. Never declares PASS or certification and never audits its own dossier.
- Subagents: one batch each, write only their own shard files, no pytest, no commits, at most 3 at a time.
- Independent auditor: separate session, role `independent-quality-auditor`, verdict on the dossier only.
- Operator: starts/approves the auditor session, decides escalations, merges the publication PR.

## Rules
R-SHA-1..5 apply (every statement about current state refers to the baseline; every historical SHA stored in full with a role label and checked for existence and ancestry; abbreviations expanded; no mixing of post-baseline evidence; identical header block in every deliverable). All Markdown deliverables are rendered from data by `tools/render.py`; hand edits are forbidden.

## Enumerations
- Evidence levels: `E0` no evidence; `E1` document claim only; `E2` E1 plus resolvable SHA and existing cited artifact; `E3` E2 plus artifact content checked against the claim; `E4` re-executed by EQO-A at the baseline (test budget).
- Severity: `BLOCKER`, `HIGH`, `MEDIUM`, `LOW`, `INFO`. BLOCKER = would invalidate an ARCH-FREEZE entry condition (roadmap invariants 22-23) or breaks a hard tier boundary in production code. HIGH = a closed stage's central claim is unsupported, or a security/tenant invariant is unproven. BLOCKER/HIGH require evidence level >= E3, otherwise the class is `UNVERIFIED_HYPOTHESIS` with `next_action: CAMPAIGN_B` and the severity is labelled "potential".
- Finding class: `CONFIRMED_DEFECT`, `EVIDENCE_DEFICIENCY`, `GOVERNANCE_INCONSISTENCY`, `CERTIFICATION_DRIFT`, `ARCHITECTURE_RISK`, `UNVERIFIED_HYPOTHESIS`, `ACCEPTED_SCOPED_LIMITATION`.
- A2 certification verdicts: `CV-VERIFIED`, `CV-SCOPED`, `EV-INCOMPLETE`, `EV-INCONSISTENT`, `HIST-REQUAL`, `CLAIM-UNSUPPORTED`, `OPEN_NOT_CERTIFIED`, `NOT-ASSESSED` (`NOT-ASSESSED` is allowed in shards only and must be 0 by W5; `OPEN_NOT_CERTIFIED` is used for stages that have no certification claim).
- A5 drift classes: `DRIFT-NONE`, `DRIFT-BENIGN`, `DRIFT-REQUALIFIED`, `DRIFT-UNREQUALIFIED`, `DRIFT-UNKNOWN`.
- `next_action`: `OWNER_DECISION`, `CAMPAIGN_B`, `CAMPAIGN_C`, `REQUALIFY`, `DOC_FIX_BY_OWNER`, `NONE`. `arch_freeze_impact`: `BLOCKS`, `CONDITIONS`, `NONE`, `UNKNOWN`.
- Roadmap status vocabulary (section 2.2): `CLOSED`, `CURRENT`, `READY FOR AUDIT`, `BLOCKED`, `QUEUED`, `PLANNED`, `FINAL / MANDATORY`; anything else is `OFF_VOCAB`.
- FRZ status vocabulary (checklist): `OPEN`, `PASS`, `BLOCKED`, `N/A — WITH EVIDENCE`; anything else is `OFF_VOCAB`.
- Stage disposition: `PARENT_ASSESSED`, `CHILD_INHERITS:<parent>`, `LINEAGE_ONLY`, `OPEN_NOT_CERTIFIED`, `PROSE_ORPHAN`, `OUT_OF_SCOPE:<reason>`. Assessment tier: `T1` (full 12 questions), `T2` (mechanical).
- SHA roles: `implementation`, `audit`, `test_execution`, `certification`, `closure_publication`, `rejected`, `superseded`, `evidence`.

## Out of scope and forbidden
No edit of any tracked file, no FRZ promotion, no stage or ticket closure, no exception acceptance, no migrations or production changes, no Docker or new vendors, no conversion of red tests to skips, no checkout/reset/stash in the main checkout, no commit other than the single publication PR in W6-P. Forbidden tools: `promote-baseline`, `--bootstrap-candidate`, `--write-baseline`, `igx_gov bootstrap|sync-adapters|ticket|before-push`, `run_regression_gate_ci.py`, `generate_roadmap_replay_x_matrices.py`, `render_*` without `--check`, `sync_agent_env.py`.

## Completion statement
Completion states only that the baseline was inventoried, evidence-graded and independently checked within the declared scope. It does not state that the platform is defect-free.
