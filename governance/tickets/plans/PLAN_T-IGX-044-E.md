# PLAN T-IGX-044-E - EQO-A remediation node PT-E04 (T-ENF-4): new IIAC corpus, artifact and delegation verifiers

Scope statement: this plan describes new files only, wires nothing into any gate, and claims no PASS, closure, FRZ promotion or independent verification. Approval state is recorded only in the ticket's Approved section and lifecycle state.

## 1. Purpose

EQO-A Campaign A found that the IIAC and ticket validators rest on string comparison (findings A2-038 part, A2-006, A6-002) and that 3 of the published IIAC manifests fail the repository's own manifest validator while staying tolerated. Plan B of the WP-7 enforcement plan (design V1) specifies two NEW verifiers for this: a corpus verifier and an artifact verifier. Decision DP-58 (RB-1 as planned, validator hardening lands before the RB-2 audit) and DP-68 (host-issued identity evidence now, legacy exceptions file, tickets untouched) ask for the checks to exist and be tested before the material validators are changed in PT-E05 (T-ENF-5).

This ticket adds ONLY new files: three read-only verifiers, one shared helper module, one allowlist of the known invalid drafts, and a corpus of positive and negative cases with their expected verdicts. It edits no existing file under `scripts/` or `tests/`, no IIAC script, no IIAC protocol text and none of the six material governance paths.

## 2. What is added

| File | Role | WP-7 design item |
|---|---|---|
| `scripts/governance/verify_iiac_manifest_corpus.py` | Enumerates every `iiac_audit_session_manifest*.json` under a root; a manifest that does not validate must be in the known-draft allowlist (path, reason, LF-normalised sha256); fails on unlisted invalid, stale or orphan allowlist entries, and on a `--certifying` manifest that is not final | V1 (1) |
| `scripts/governance/verify_iiac_manifest_artifacts.py` | For one manifest: SHAs exist as commits and are ordered, digests are bound to the bytes of the files they name, placeholders and non-certifying verdicts cannot pass as final, role separation beyond string inequality. Profiles `final` (default, all errors) and `legacy` (published drafts are reported, not rejected) | V1 (2), part of V2 (role checks only) |
| `scripts/governance/verify_iiac_delegation_refs.py` | For tickets: every delegate approval line, in the documented two-line form (`Approved: operator-delegate @ <ts>` with `Delegation: <path>@<sha>` directly on the next line, `governance/tickets/README.md` item 3) and in the older single-line form (`Approved: operator-delegate <label> @ <ts> \| Delegation: <path>@<sha>`), must resolve: commit exists, path exists at it, the commit is an ancestor of HEAD (a record that is only in the history of HEAD but not of the approval commit is a warning), the record names the ticket (exact id, a range such as `T-IGX-041-A .. T-IGX-041-I`, or its `T-IGX-NNN-*` family) and, when a label is present, the delegate; a delegate label equal to the implementer fails. Without a label (the documented form has none) delegate != implementer cannot be proven and is reported as `DELEGATE_IDENTITY_UNKNOWN` (a warning, an error with `--require-implementer`) | resolver design of V3/V4, reusable by PT-E02 and PT-E05 |
| `scripts/governance/iiac_verifier_common.py` | Shared read-only helpers (git wrapper limited to `cat-file`, `merge-base`, `rev-parse`, `show`, `log`, `ls-tree`, `ls-files`, which refuses any caller-supplied revision, ref or `rev:path` operand that starts with `-` before it reaches git; finding type; checks) | - |
| `scripts/governance/iiac_known_draft_manifests.v1.json` | The 3 published invalid drafts, each with reason and digest; they are listed, never edited (the evidence index, the preflight and the debt register cite their bytes) | V1 (1), A2-006 handled without touching the bytes |
| `tests/unit/governance/test_iiac_new_verifiers.py` | Runs every corpus case against a throw-away git repository, plus replay over the real repository and scope guards | - |
| `tests/fixtures/iiac_verifier_corpus/` | `cases.json` (59 cases: 51 negative, 8 positive), `base_manifest.json.tmpl`, evidence and delegation fixtures, README | - |

Common contract of the three verifiers: stdlib only (plus read-only reuse of `iiac_common.validate_manifest_dict`, so "validates" means what the existing validator means), write nothing, exact exit codes: 0 PASS, 1 FAIL (at least one error finding), 2 usage or environment error (missing file, unusable repository, bad allowlist schema, bad flag). Findings carry a stable code (for example `DIGEST_MISMATCH`, `SHA_ORDER`, `DELEGATION_UNRESOLVED`); a `warn` finding never changes the exit code.

## 3. Not material, not wired, not imported (facts to be re-checked at the audited SHA)

- `iiac_activation_gate.py:40-49` at `origin/development` `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2` (the branch base; re-read unchanged at `f98a0ce959ec183430576d3aaed46c8deadd32fe`, which differs from the base only in other tickets, plans and documents, not in `scripts/` or `tests/`) lists exactly six `MATERIAL_GOVERNANCE_PATHS`: `verify_iiac_policy_activation.py`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py` and the IIAC protocol document. None of the files of this ticket is on the list (`test_new_files_are_not_material_and_not_imported_by_the_activation_gate` asserts it).
- The gate calls only `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py` by name (`iiac_activation_gate.py:310-316`) and compares changed paths with the list (`:342`) and `PRODUCT_PATH_PREFIXES` (`iiac_common.py:22`, `intergrax/ agents/ applications/`). No material file imports or names a new module (asserted by the same test), so the activation record check, the stale-audit check (ACT-N11 rule i) and the product-path check (rule ii) are unchanged by this ticket. `scripts/` and `tests/` are not product prefixes.
- The new verifiers import `iiac_common` (read only). A later change of `iiac_common.py` (PT-E05) changes what "validates" means for them; that is intended and is a PT-E05 audit item.
- Not wired: no workflow, hook, `igx_gov` command, gate or pytest collection hook calls the new verifiers. Wiring them (CI, `igx_gov`, activation companion calls, `MANDATORY_IIAC_GOVERNANCE_PATHS` in `iiac_g0_inventory.py`) is PT-E05 / T-ENF-5, a material governance change that needs SCOPE and the IIAC re-audit. This plan does not make it.
- The new files appear in the next G0 inventory regeneration as governed paths; the committed matrix is bound to an earlier `inventory_source_sha`, so its completeness check is unaffected (existing `test_iiac_g0_inventory.py` is part of the regression run).

## 4. Measured behaviour at the head (read-only replay, 2026-11-16)

- Corpus verifier on the real repository with the published allowlist: PASS. 41 manifest files matched (including the example manifest; the dossier counted 39 at its baseline); 3 invalid, all listed (the same three drafts the campaign named); without the allowlist the verifier fails with exactly those 3 `MANIFEST_INVALID`. Legacy-profile warnings over the whole corpus (counts of findings, not files): `PLACEHOLDER` 37, `VERDICT_NOT_CERTIFYING` 18, `DIGEST_UNBOUND` 52, `DIGEST_ARTIFACT_MISSING` 51, `DIGEST_MISMATCH` 8, `SHA_ORDER` (strict chain) 8, `ROLE_AGENT_EQUAL` 1 (ENT-AUDIT-X certification: implementer and certifier share the label `certification-governance-agent`). These are reported, not rejected, because published drafts are superseded in registers and never edited. They are inputs for PT-E05 and for the register, not findings closed here.
- Core SHA order (`audited <= audit_execution`, `test_execution <= audit_execution`) holds for every published manifest; the strict chain `audited <= test_execution <= audit_execution <= evidence <= certification` is violated by activation-type and some historical manifests, so it is an error only in profile `final` (new manifests), a proposal PT-E05 may adjust.
- Delegation verifier over the 146 `governance/tickets/T-IGX-*.md` at the merged head (re-run after the audit finding that the first version read only the single-line form): 93 delegate lines, 84 in the documented two-line form, 8 in the single-line form, 1 with a Delegation line that is not directly after (`T-IGX-035-F`); 21 tickets have no Approved line (out of scope here). 17 tickets carry an error finding: `T-IGX-030-A` (unpinned `.tmp/` path), `T-IGX-034-B` (path absent at the pinned commit), `T-IGX-035-E` (free text), `T-IGX-035-F` (not adjacent), and 13 whose record does not name the ticket (`T-IGX-036-Y`, `036-Z`, `041-M`, `041-N`, `041-O`, `050-A2`, `050-B`, `050-D`, `050-F`, `050-G`, `050-H`, `050-I`, `050-J`: the standing-delegation or register file they cite lists other ids). Warnings: 84 `DELEGATE_IDENTITY_UNKNOWN` (no label in the documented form), 89 `IMPLEMENTER_UNKNOWN` (no `Implementer session:` line in any ticket), 23 `DELEGATION_OUTSIDE_APPROVAL_HISTORY` (the record was published on another branch or later than the approval commit). `T-IGX-044-A`, `046-B` and `042-A` style two-line tickets resolve with warnings only. Counts are re-computed at the audited head, not frozen here; treating the failing tickets as legacy exceptions is DP-68 work in PT-E02, this ticket creates no exceptions file and rewrites no ticket.

## 5. Negative tests

All in `tests/unit/governance/test_iiac_new_verifiers.py` over `tests/fixtures/iiac_verifier_corpus/cases.json`. Every negative case names the weakness of the existing validators it targets (the test fails on an empty `weakness`) and asserts the exit code and the finding code. 59 cases: 19 manifest-artifact, 29 delegation (11 of them for the two-line form), 11 corpus; 51 negative, 8 positive.

| Verifier | Negative cases (weakness targeted) |
|---|---|
| artifacts | forged digest (N-11 length only); 16-char digest; digest with no artifact path; artifact path missing at the cited commit; SHA that is not a commit object (syntax only); test SHA on a side branch and test SHA after audit execution (only `audited <= audit_execution` is checked); evidence before the audit; `PENDING` identity; all-zero SHA; verdict FAIL and CONDITIONAL used as certifying; implementer equals auditor session (N-01 stays detected); same session with other case or spacing; composite implementer id containing the auditor id; equal agent ids (checked by the gate only) |
| corpus | unlisted invalid draft; unparseable JSON (the N-09 guard skips it); allowlisted draft changed after listing; allowlist entry for a missing file; entry for a manifest that validates now; entry without reason; unknown allowlist schema (exit 2); FAIL-verdict manifest named certifying; valid-looking manifest citing a missing commit; role overlap inside the corpus |
| delegation | single-line and two-line forms: path missing at the pinned commit; SHA not a commit; free-text reference; unpinned path; `.tmp/` path; delegate line without any reference; two-line form with the Delegation line missing or not directly after the Approved line; delegate equals implementer (exact, normalised, composite); delegation commit on a side branch (not an ancestor); record names another ticket; delegate label only a substring of the designated session (`signer-one` vs `signer-one-b`); delegate not in the record; personal `operator @` line pointing at a delegation, inline or on the next line; timestamp without offset; implementer or delegate identity unknown under `--require-implementer` |

Positive cases: a fully valid manifest (`final`), a draft under `legacy`, digest supplied by `--artifact`, a valid manifest next to a listed draft, a valid delegate line in each form, a personal line, an unknown implementer (warning only); replay over published two-line tickets (`T-IGX-044-A`, `046-B`, `042-A`) and single-line tickets (`030-B`, `030-A`, `034-B`, `035-E`); every caller-supplied git operand that starts with `-` is refused (exit 2 at the command line). Further guards: exact exit codes through `python -I` (0, 1, 2), the verifiers leave the working tree unchanged on the real repository, the published allowlist is exactly the three known drafts, new files are not on the material list and not named by any material file, the scripts import only the standard library and call no filesystem writer or mutating git subcommand.

## 6. Write set

The only paths the pull request may touch (one per line):

```text
scripts/governance/iiac_verifier_common.py
scripts/governance/verify_iiac_manifest_corpus.py
scripts/governance/verify_iiac_manifest_artifacts.py
scripts/governance/verify_iiac_delegation_refs.py
scripts/governance/iiac_known_draft_manifests.v1.json
tests/unit/governance/test_iiac_new_verifiers.py
tests/fixtures/iiac_verifier_corpus/README.md
tests/fixtures/iiac_verifier_corpus/base_manifest.json.tmpl
tests/fixtures/iiac_verifier_corpus/cases.json
tests/fixtures/iiac_verifier_corpus/evidence/report.md
tests/fixtures/iiac_verifier_corpus/evidence/scope.json
tests/fixtures/iiac_verifier_corpus/evidence/matrix.txt
tests/fixtures/iiac_verifier_corpus/evidence/findings.json
tests/fixtures/iiac_verifier_corpus/delegations/DELEG-GOOD.md
tests/fixtures/iiac_verifier_corpus/delegations/DELEG-OTHER.md
tests/fixtures/iiac_verifier_corpus/delegations/DELEG-SIDE.md
tests/fixtures/iiac_verifier_corpus/delegations/DELEG-SUBSTR.md
governance/tickets/T-IGX-044-E.md
governance/tickets/plans/PLAN_T-IGX-044-E.md
```

Every path is new. Nothing under `intergrax/`, `agents/`, `applications/`, `.github/`, `docs/` is touched, and no existing file under `scripts/` or `tests/`. Not touched in particular: `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, `verify_iiac_policy_activation.py`, `iiac_common.py`, `iiac_activation_gate.py`, `verify_od_ent_02_compensating_control.py`, `verify_plan_approval.py`, `verify_ticket_lifecycle.py`, the IIAC protocol, the roadmap, the FRZ checklist.

## 7. Test contract

One process at a time, under the shared pytest lock, at most 15 minutes:

- One `uv run pytest tests/unit/governance/test_iiac_new_verifiers.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_g0_inventory.py -o cache_dir=<scratch dir>` (new tests plus the existing IIAC validator tests as regression). Results are recorded honestly in the pull request, including pre-existing failures if any; the 3-clone suite `test_evol_x_real_repo_fresh_clone.py` is not run (it exercises validators this ticket does not change).
- One `ruff check` and one `ruff format --check` over the new Python files.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the 19 paths above, all added.

## 8. Independent audit (AUD-ENF-2) and review notes

The plan needs an independent audit at the exact pull-request head SHA by a session that did not implement or review it (WP-12 node PT-Q02; WP-7 package AUD-ENF-2, which feeds the IIAC re-audit). Suggested scope for that auditor: mutate each check (drop the digest comparison, the ancestry test, the placeholder test, the normalised role comparison, the allowlist digest comparison, the approval-commit ancestry) and confirm a named test fails; confirm read-only behaviour on a clean checkout; replay the corpus verifier and the delegation verifier on the head; confirm the six material paths and the five IIAC scripts are byte-identical to `origin/development`. The implementer does not sign or certify anything.

Operator decisions on the open points, recorded 2026-11-17 (addendum 3 of `OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md`): (a) the strict SHA chain is the rule for new manifests, in profile `final` only (`legacy` keeps the core chain as errors and the strict chain as warnings); (b) the known-draft allowlist stays in `scripts/governance/`, because it must travel with the verifier; (c) the delegation verifier stays separate; whether PT-E02 or PT-E05 folds it in is decided with PT-E05. Still open: (d) whether the supersession record for the three drafts (A2-006) goes into the registers (WP-6), which this ticket does not touch.

## 9. Risks and rollback

- Risk: the verifiers are untested in CI and not wired, so they protect nothing until PT-E05; the plan says so rather than implying enforcement.
- Limits of the delegation matching (documented in the verifier): tokens are case-insensitive whole tokens (`signer-one` does not match inside `signer-one-b`); the scope check cannot tell a record that merely mentions the ticket id from one that delegates it; delegate identity is not machine-readable when the line carries no label.
- Risk: allowlist digests are LF-normalised sha256 of the published bytes; if a draft is ever edited, `ALLOWLIST_STALE_DIGEST` fires by design.
- Risk: the corpus builds a git repository in the pytest temp directory; no network, no tracked path is written.
- Rollback: revert the single pull request. No data, configuration, workflow or product path is affected.

## 10. Closure condition

The pull request is merged after review and the AUD-ENF-2 audit, and the ticket records the merge SHA. No finding is marked closed or verified by this ticket; A2-038 (part), A2-006 and A6-002 stay open in the register until PT-E05 / PT-E02 land and are independently audited.
