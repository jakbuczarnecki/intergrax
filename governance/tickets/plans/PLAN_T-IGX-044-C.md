# Plan T-IGX-044-C - EQO-A node PT-E02 (T-ENF-2): ticket-approval validators, legacy-exceptions file, ticket-governance job, Ref gate CI mode

Sources: WP-7 plan B, section V4, finding EQOA-A6-002 (and the A2-038 identity part that lives in tickets); Operator decisions DP-68, DP-10, DP-58, DP-66 (decision sessions 1 and 6 of 2026-11-16); node row `PT-E02` of the remediation ticket draft. Baseline of this plan: `origin/development` at `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2`. This plan is not a certification verdict and promotes no FRZ status.

## 1. Problem

- `verify_plan_approval.py` and `verify_ticket_lifecycle.py` accept any line that starts with `Approved:` (`_APPROVED_RE = ^Approved:\s*.+$`). A delegate line is therefore only as good as its `Delegation:` reference, and nothing resolves that reference.
- Neither validator, nor the commit Ref gate, runs over real tickets in CI: the only callers are unit tests; the Ref gate is client side with an environment skip (`INTERGRAX_SKIP_TICKET_REF_GATE`) and silently passes when `git log` fails.
- 4 delegate approvals carry a reference that does not resolve (T-IGX-030-A, 034-B, 035-E, 035-F); 19 files under `governance/tickets/T-*.md` have no `Approved:` line (one of them is an audit report, not a ticket).
- Independence of the delegate from the implementer is a string comparison in a handoff, not a field of the ticket.

## 2. Design (additive: the default of every existing validator is unchanged)

1. New library `scripts/governance/ticket_approval.py`: strict grammar, delegation resolver, legacy-exceptions loader.
2. New CLI `scripts/governance/verify_ticket_approvals.py` with three modes: `--since <base>` (tickets added or whose approval block or lifecycle state changed since the merge-base; used by CI), `--all` (read-only replay over every tracked ticket, the input of node PT-V06) and `--ticket <path>`.
3. Grammar (strict mode only): exactly one `Approved:` line, either `Approved: operator @ <ISO8601 with offset>` or `Approved: operator-delegate @ <ISO8601 with offset>`; the delegate form needs, on the next two lines, `Delegation: <repo-relative path>@<sha40>` and `Delegate session: <host session id>`, and the ticket needs one `Implementer session: <host session id>` field. Host session ids have the form `local_|cloud_|remote_<uuid>` (DP-68: host-issued session id as identity evidence now; per-role keys later; the label `operator-delegate` is kept).
4. Delegation resolver (read-only git): the path is repo-relative and not under `.tmp/` or `.git/`; the sha is a commit; the record is a tracked file at that sha (`git cat-file -e`); the sha is a strict ancestor of the commit that introduces the approval block; the record text names the ticket (exact id, a letter range such as `T-IGX-041-A..I`, or `T-IGX-NNN-*`); the delegate session differs from the implementer session.
5. Lifecycle rule: in states `plan_approved`, `in_progress`, `quality_gate`, `ready_for_audit`, `closed` (and any unknown state) an approval block is required; in `draft` and `plan_pending` it may be absent, and if present it must be valid.
6. Legacy exceptions: new file `governance/tickets/legacy_approval_exceptions.json` with the 19 files without an approval line and the 4 unresolvable references, each with reason, finding and decision reference. Existing tickets are not edited. An exception is never silent: every applied exception is printed as `[EXCEPTION]` and counted in the summary; entries of kind `unresolvable_delegation_reference` are bound to the sha256 of the exact approval lines (a changed line voids the entry); in `--since` mode the file is read from the merge-base and applies only to a ticket that already existed there, so a pull request cannot excuse its own ticket; in `--all` mode a stale entry is an error.
7. Ref gate CI mode: `verify_commit_ticket_refs.py --ci` ignores the skip environment variable (warning printed), treats an empty range or a failing `git log`/`git show` as an error, and requires the referenced ticket to be tracked at HEAD. Without `--ci` the behaviour is byte-for-byte the previous one.
8. CI: new workflow file `.github/workflows/ticket-governance.yml` with one job `ticket-governance` (no `paths:` filter, no job `if:`), running `verify_ticket_approvals.py --since origin/<base>` and `verify_commit_ticket_refs.py --ci --mode range --rev-range origin/<base>..HEAD`. A new file is used on purpose so that the job does not touch `unit-tests.yml` and `enterprise-audit*.yml` (open pull request 616). The job is not a required check.

## 3. Write set

governance/tickets/T-IGX-044-C.md
governance/tickets/plans/PLAN_T-IGX-044-C.md
governance/tickets/legacy_approval_exceptions.json
scripts/governance/ticket_approval.py
scripts/governance/verify_ticket_approvals.py
scripts/governance/verify_commit_ticket_refs.py
tests/unit/governance/test_ticket_approval_validator.py
tests/unit/governance/test_ticket_ref_ci_mode.py
.github/workflows/ticket-governance.yml

Excluded on purpose and checked at `origin/development`: the six `MATERIAL_GOVERNANCE_PATHS` of `scripts/governance/iiac_activation_gate.py` (`verify_iiac_policy_activation.py`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, the IIAC protocol document), `iiac_g0_inventory.py`, `verify_iiac_g0_inventory_completeness.py`, `generate_iiac_authority_impact_matrix.py`, `verify_od_ent_02_compensating_control.py`. Everything in this write set is outside `intergrax/`, `agents/`, `applications/` product paths for ACT-N11 purposes except `scripts/` (a commit-Ref scope only, not an ACT-N11 path). The IIAC-side identity items (provenance block in manifests, operator countersignature line, delegate resolver in the activation gate) change material files and belong to PT-E05 (T-ENF-5); they are not done here.

## 4. Collision check (read at `origin/development` `2ab57e36d`)

- T-IGX-041-P writes `intergrax/rag/graph/**` and one new test under `tests/unit/rag/graph/`; T-IGX-041-Q writes two files under `tests/qualification/ebh_7/`. No overlap with the write set.
- Open pull requests (8 at read time: 612, 613, 614, 615, 616, 617, 618, 619): the only files they share with the directories of this write set are their own new `governance/tickets/plans/PLAN_T-IGX-0xx-x.md` files (distinct names) and, for pull request 616, `.github/workflows/enterprise-audit.yml`, `enterprise-audit-nightly.yml` and `unit-tests.yml`, which this plan does not touch (the new workflow is a new file). No file of this write set is touched by another open pull request.

## 5. Gate-meaning classification

| Part | Changes what a gate accepts or refuses? | Ruled in by | In the first commit |
|---|---|---|---|
| New strict verifier and library | New tool; no existing gate calls it | DP-68, DP-58 | yes |
| Legacy-exceptions file | Data for the new tool only | DP-68 (quoted: "legacy exceptions file for the unresolvable delegation references and tickets without an Approved line, tickets untouched") | yes |
| `verify_commit_ticket_refs.py --ci` | Refuses more, but only when `--ci` is passed; default unchanged | DP-66/DP-58 (enforcement hardening lands before RB-2) read with plan B | yes, opt-in |
| New workflow job `ticket-governance` | Adds a check that can show red on a pull request; not required, so it blocks no merge | plan row PT-E02 names the job | yes, non-required |
| Making `ticket-governance` a required check | Yes (Plan A Stage 4, node PT-S05) | not ruled in; Operator authorisation AR-49 | NO, not part of this ticket |
| Wiring the strict verifier into `igx_gov ticket` (local pre-push) | Yes: a developer push that was accepted would be refused | not ruled in by DP-68/DP-66 | NO: needs the Operator's personal ruling |
| Flipping the default of `verify_plan_approval.py` / `verify_ticket_lifecycle.py` to the strict grammar | Yes: tickets accepted today would be refused | not ruled in | NO: needs the Operator's personal ruling; both files stay unchanged |
| README statement "not enforced by tooling" (`governance/tickets/README.md`) | Documentation | n/a | NO: outside the write set; follow-up docs change |
| Operator countersigned handoff and per-role identity in IIAC manifests | Yes, material files | DP-68 for the design | NO: PT-E05 |

## 6. Negative tests

All in `tests/unit/governance/test_ticket_approval_validator.py` and `test_ticket_ref_ci_mode.py`, each on a throw-away git repository under `tmp_path` (the real repository is never touched):

- free text after `Approved:`; missing offset; invalid date; trailing comment; foreign label; two approval lines; personal form carrying delegation lines;
- delegate approval without a `Delegation:` line (must fail in the new mode); without a `Delegate session:` line; with a malformed session id;
- delegation path under `.tmp/`, absolute, `..`, `.git/`; record absent at the cited sha; unknown sha; sha not an ancestor of the approval commit; record that does not name the ticket (and positive cases: range and wildcard notation);
- delegate session equal to the implementer session; missing or malformed `Implementer session:`;
- post-plan states (all five) without an approval line fail; `plan_pending` with the `PENDING.` placeholder passes; missing lifecycle state fails;
- unreadable base fails closed; CLI exit codes;
- legacy exceptions: accepted only through the file and reported; the same ticket fails without the entry; a pull request cannot add its own exception; a new ticket cannot use an exception that exists at the merge base; an exception does not excuse an invalid approval line; the digest-bound entry is void when the approval lines change; schema errors fail closed; stale entries fail in `--all`; the replay summary counts every excepted ticket;
- Ref gate `--ci`: product commit without Ref fails; Ref to an untracked ticket fails only in CI mode; skip environment ignored in CI mode and still honoured in default mode; unreadable range and empty range are errors in CI mode only; non-product commit needs no Ref.

Regression nodes (verified to exist at `origin/development`): `tests/unit/governance/test_agent_role_routing.py::test_verify_ticket_lifecycle_valid_ticket`, `::test_verify_ticket_lifecycle_rejects_missing_approved`, `tests/unit/governance/test_dcp_control_plane_registry.py::test_plan_digest_stale_fails_verification`, and every test of `tests/unit/governance/test_dcp_commit_ticket_refs.py`. They pass because the legacy validators and the default Ref gate are not changed.

## 7. Replay design for node PT-V06 (not executed by this ticket)

`python -I scripts/governance/verify_ticket_approvals.py --all --json .tmp/session/pt-v06/replay.json` at the exact SHA chosen by PT-V06, read-only, one process. Expected result to be judged by PT-V06: the 23 listed exceptions are applied and counted; every other failing ticket is a finding of the replay, because the strict grammar is stricter than the legacy practice (for example approval lines with a trailing comment, an inline `| Delegation:` form, labels such as `operator-delegate hist-sign-plans`, and two tickets whose line is a decision list). PT-V06 decides, with the Operator, whether such tickets are added to the exceptions file by a separate change or stay open findings. This ticket does not claim that the replay is clean.

## 8. Honest limits

The verifier proves form, existence, ancestry and textual scope of the record. It cannot prove who typed a line or that a session id was not copied; independence of delegate from implementer stays `process_enforced`, as in the roadmap and the A2-038 finding, until per-role keys exist (DP-68). The `Implementer session:` and `Delegate session:` values are self-asserted by their writers.

## 9. Independent audit

The ticket needs the independent G6 audit `AUD-ENF-1` (node `PT-AUD-ENF1`, covering T-ENF-1, T-ENF-2, T-ENF-3): a separate session, the implementer excluded, which re-runs the negative tests and the legacy-exception replay at the exact SHA. The implementer does not declare PASS.

## 10. Test contract and rollback

Run under the shared pytest lock, one process: `uv run pytest tests/unit/governance/test_ticket_approval_validator.py tests/unit/governance/test_ticket_ref_ci_mode.py tests/unit/governance/test_dcp_commit_ticket_refs.py tests/unit/governance/test_agent_role_routing.py tests/unit/governance/test_dcp_control_plane_registry.py -q -p no:cacheprovider`, then `uv run ruff check` and `uv run ruff format --check` on the changed Python files. Rollback: revert the merge commit (scripts, tests, one new workflow and one data file; no product effect, no GitHub setting).
