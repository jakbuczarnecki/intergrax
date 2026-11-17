# PLAN T-IGX-044-F - EQO-A remediation node PT-E05 (T-ENF-5): IIAC validator hardening on the material governance paths

Scope statement: this plan edits five of the six `MATERIAL_GOVERNANCE_PATHS` and adds one test file. It changes what the IIAC validators and the activation gate accept for NEW (v2) manifests and for NEW activation records, wires the PT-E04 verifiers into the activation gate, and claims no PASS, closure, FRZ promotion or independent verification. Approval state is recorded only in the ticket's Approved section and lifecycle state.

## 1. Purpose

EQO-A Campaign A found that IIAC independence rests on string comparison and role labels (A2-038), that three published manifests fail the repository's own validator while staying tolerated (A2-006, handled in PT-E04 without touching their bytes), and that a delegated operator line is only a format check (A2-038 operator line). Plan B of the WP-7 enforcement plan (`WP-7/PLAN.md` V1, V2, V3; ticket row T-ENF-5; decisions DP-58 and DP-68) specifies the hardening of the validators themselves. PT-E04 (T-IGX-044-E, merged as PR #623, `a8b325c06d60b01ab88b7d3e61bb4763095eb3d8`) added the corpus, artifact and delegation verifiers as new, unwired files and left the wiring and the strictness of the existing validators to this ticket.

Design rule that follows from the Operator decision (a) of 2026-11-17 (addendum 3 of `OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md`): every new strictness applies to NEW manifests (marked v2) and to NEW activation records only. A manifest without the v2 marker is a v1 manifest and keeps exactly today's verdict. Measured at `origin/development` `2727a2c2364a97492202b931375772a400279a9b`: 41 manifest files, none carries any schema or state marker today, 38 validate and 3 (the known drafts) do not; 20 of the 41 contain a placeholder-like token in an identity field (regex `HISTORICAL_UNRECORDED|PENDING|TBD|UNKNOWN|TODO`, deliberately broad), so placeholder rejection cannot apply to v1.

## 2. What changes

| File | Change (all behind the v2 marker unless stated) | WP-7 item |
|---|---|---|
| `scripts/governance/iiac_common.py` (165 lines) | `validate_manifest_dict`: (i) session ids are compared after normalisation (strip, collapse whitespace, case-fold) and a session id may not equal any agent label of another role; applies to v1 too (measured: 0 of 41 published manifests collide after normalisation, 1 has equal raw agent labels, which stays a warning for v1 and becomes an error for v2); (ii) v2 fields `manifest_schema_version` (`v2`) and `manifest_state` (`draft` or `final`), any other value is an error; (iii) v2 final: verdict must be `PASS` or `PASS_SCOPED`, no placeholder strings in any identity, SHA or digest field, `identity_provenance` required per role; (iv) new helper `resolve_delegation_ref(repo, path, sha, at)` (read-only git, commit exists, path exists at the commit, commit is an ancestor of `at`, record names the stage and a delegate session). `PRODUCT_PATH_PREFIXES` is unchanged; a second constant `GOVERNANCE_WRITE_SURFACE` (`scripts/governance/`, `tests/unit/governance/`, `docs/project/maintainers/qualification/`) is added for v2 | V1 (3), V2 (1)(4), V3 |
| `scripts/governance/verify_iiac_audit_manifest.py` (82 lines) | New arguments `--profile final\|legacy` (default `final` for v2, ignored for v1); v2 final: strict SHA chain `audited <= test_execution <= audit_execution <= evidence <= certification` through the PT-E04 rules, `handoff_record_ref` (`path@sha`) required and the operator countersignature line checked (section 4.3); an unknown schema marker exits 1 | V1 (2)(3), V2 (2) |
| `scripts/governance/verify_iiac_role_separation.py` (66 lines) | v2: product-diff failure is the default and `--allow-product-diff` is the explicit opt-out (the existing `--strict-product-diff` stays accepted and is a no-op for v2); the write surface is `PRODUCT_PATH_PREFIXES` plus `GOVERNANCE_WRITE_SURFACE` between the audited and the audit-execution SHA; host receipt uniqueness across roles and, with `--corpus-root` (default: the qualification root), across stages. v1: unchanged apart from the normalised session comparison | V2 (1)(3) |
| `scripts/governance/iiac_activation_gate.py` (578 lines) | (i) `_companion_errors` (`:305-326`, today two subprocess calls) gets two more entries, `verify_iiac_manifest_corpus.py` and `verify_iiac_manifest_artifacts.py` (profile `legacy` for a record whose manifest is v1); (ii) `MATERIAL_GOVERNANCE_PATHS` (`:40-49`) gains the four files those calls depend on (section 3); (iii) new record field `operator_line_kind` (`personal` or `delegate`): a record without it is read as today (personal); a new record must carry it; for `delegate` the operator decision record must resolve through `resolve_delegation_ref` and the personal-form line may not point at a delegation file. The ACT-N11 rules (`:567-576`) are not changed | V3, V1 wiring |
| `docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` (122 lines) | Defines the v2 marker, `manifest_state`, `identity_provenance`, the `Countersigned: operator @ <ts> manifest-digest <sha256>` line, the operator line kinds, and the honest limit: without per-role signing keys independence stays `process_enforced` and every verdict keeps the scoped label "internal agent certification" | V2 (5) |
| `tests/unit/governance/test_e05_role_separation_identity.py` (new) | All negative tests and controls of section 5 in one file, throw-away git repositories in the pytest temp directory, no network, no tracked path written | NT-ROW-010 |

Not changed: `scripts/governance/verify_iiac_policy_activation.py` (the sixth material path; the CLI wrapper needs no change because the gate carries the logic), the v1 manifest schema file (`additionalProperties` is `true`, v2 fields need no schema edit), the handoff template, `iiac_g0_inventory.py`, every workflow, hook and `igx_gov` command, every ticket validator, every product path, every published manifest, the activation record file, and any existing test.

## 3. Material paths, ACT-N11 and the re-pin

- `iiac_activation_gate.py:40-49` lists six material paths: `verify_iiac_policy_activation.py`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py` and the protocol document. All six are byte-identical at `origin/development` `2727a2c23` and at the recorded audit SHA `a73d15ab341b30e9748aad03894f6faa75ad4e09` (empty diff). This ticket edits five of them (not `verify_iiac_policy_activation.py`).
- Consequence, stated without a workaround: after this ticket merges, `_material_changes(audit_sha, head)` reports `ACT-N11 stale audit after material governance change` for each edited path until the single RB-2 re-pin (new IIAC audit of the integrated head, new activation record). The same re-pin also covers the product-path drift that the other train tickets cause (`:570-576`). No change here is meant to make the gate pass without the re-pin.
- Extending `MATERIAL_GOVERNANCE_PATHS` is itself a change of the gate's meaning and is made in the same edit as the wiring: the new entries are `scripts/governance/iiac_verifier_common.py`, `scripts/governance/verify_iiac_manifest_corpus.py`, `scripts/governance/verify_iiac_manifest_artifacts.py` and `scripts/governance/iiac_known_draft_manifests.v1.json` (the four files the wired calls execute or read). The delegation verifier is not wired here (it checks tickets, which `ticket-governance` of PT-E02 already covers).
- The v1 path is kept so that the existing record keeps verifying its own companion checks when the re-pin is prepared (control C2).

## 4. Design

### 4.1 Manifest v2 and strict default (V1)

A manifest is v2 when it carries `manifest_schema_version: "v2"`. v2 adds `manifest_state` (`draft` or `final`), `identity_provenance` and `handoff_record_ref`. A `draft` may carry placeholders and may not be used as certifying (the corpus verifier of PT-E04 already fails a non-final manifest under `--certifying`). A `final` is held to the PT-E04 artifact rules as errors, not warnings: SHAs exist and follow the strict chain (rule for new manifests, Operator decision (a)), the four digests equal the bytes of the files they name, verdict is `PASS` or `PASS_SCOPED`, no placeholder strings. v1 manifests are never reinterpreted.

### 4.2 Identity and provenance (V2)

`identity_provenance` per role: `{issuer, session_uuid, started_at_utc, host_receipt_ref}`, `host_receipt_ref` = `path@sha` of a tracked export of the host session id. Checked: the file exists at the commit, the commit is an ancestor of the audit-execution SHA, the file contains the `session_uuid`, the `session_uuid` equals the role's session id after normalisation, no `session_uuid` or receipt repeats across the three roles, and none repeats in another v2 manifest of the corpus. The check proves consistency of recorded identifiers, not who typed them: independence stays `process_enforced` (open question Q-E05-05 for signed commits).

### 4.3 Countersignature and operator line kinds (V2, V3)

The handoff record named by `handoff_record_ref` must contain a line `Countersigned: operator @ <ts> manifest-digest <sha256>` whose digest equals the LF-normalised sha256 of the manifest bytes; presence and digest are checked mechanically, authorship is not. The operator line of an activation record is `personal` (`Approved IIAC policy activation <version>: operator @ <ts>`) or `delegate` (`operator-delegate @ <ts>` plus a `Delegation: <path>@<sha40>` pointer); the gate resolves the pointer with `resolve_delegation_ref` and fails when the record names another stage, is not an ancestor of HEAD, or names the implementer as the delegate. Open question Q-E05-03 asks whether this resolver stays a small copy in `iiac_common.py` (default) or reuses `ticket_approval.resolve_delegation` (PT-E02).

### 4.4 Wiring (V1, V3)

Through the gate only (section 2). No workflow, hook or `igx_gov` change is part of this ticket; a CI job for the corpus verifier is a separate non-material ticket (open question Q-E05-02).

## 5. Findings of the PT-E04 verifiers that this ticket consumes

Legacy profile over the whole corpus at the PT-E04 head (re-checked: no manifest or verifier file changed between `401f896ee` and `2727a2c23`). Finding counts: PLACEHOLDER 37, VERDICT_NOT_CERTIFYING 18, DIGEST_UNBOUND 52, DIGEST_ARTIFACT_MISSING 51, DIGEST_MISMATCH 8, SHA_ORDER 8, ROLE_AGENT_EQUAL 1.

| Finding | Disposition in this ticket |
|---|---|
| PLACEHOLDER 37, VERDICT_NOT_CERTIFYING 18 | error for v2 `final` (4.1); v1 stays a warning; the three drafts stay in the allowlist |
| DIGEST_UNBOUND 52, DIGEST_ARTIFACT_MISSING 51, DIGEST_MISMATCH 8 | error for v2 `final`; v1 stays a warning; no published manifest is edited |
| SHA_ORDER 8 | the core order stays an error in both profiles, the strict chain becomes an error for v2 `final` only |
| ROLE_AGENT_EQUAL 1 | error for v2; for v1 one published manifest (`ENT-AUDIT-X/certification`) keeps a warning; no byte of it changes |

None of these counts is a defect list to repair here. They are the measured weight of v1 history that a global strict default would have rejected, which is why strictness is gated on v2.

## 6. Negative tests and controls

All in `tests/unit/governance/test_e05_role_separation_identity.py`; each test asserts the exit code or error text and names the weakness it targets. `NT-E05-01` is the RW-PLAN row NT-ROW-010 (finding A2-038).

| Group | Tests |
|---|---|
| Role and identity (12) | NT-E05-01 sessions that differ only in case, whitespace or a label are rejected by `verify_iiac_role_separation`; NT-E05-02 one id used under two role labels; NT-E05-03 v2 identical `session_uuid` across roles; NT-E05-04 v2 placeholder identity (`HISTORICAL_UNRECORDED...`, `PENDING_CERTIFIER`); NT-E05-05 v2 without `identity_provenance`; NT-E05-06 host receipt duplicated across stages; NT-E05-07 receipt not at an ancestor or without the session id; NT-E05-08 v2 auditor commit under `scripts/governance/` between the two SHAs; NT-E05-09 v2 product diff without `--allow-product-diff`; NT-E05-10 missing countersignature; NT-E05-11 countersignature digest mismatch; NT-E05-12 a v2 manifest listed in the known-draft allowlist to bypass placeholder rejection |
| Manifest (6) | NT-E05-13 unknown schema marker or `manifest_state` fails closed; NT-E05-14 draft used as certifying; NT-E05-15 final with FAIL or CONDITIONAL; NT-E05-16 final with a placeholder string; NT-E05-17 final with a missing SHA object or a strict-chain violation; NT-E05-18 final with a digest not bound to file bytes |
| Gate and delegation (6) | NT-E05-19 delegate line without a record; NT-E05-20 record not an ancestor of HEAD; NT-E05-21 record naming another stage; NT-E05-22 personal-form line pointing at a delegation file; NT-E05-23 new record without `operator_line_kind`; NT-E05-24 delegate equal to the implementer |
| Wiring (3) | NT-E05-25 an unlisted invalid manifest makes the gate companion fail; NT-E05-26 a changed known-draft digest makes it fail; NT-E05-27 each wired file is a member of `MATERIAL_GOVERNANCE_PATHS` |
| Controls (2) | C1 replay: the 41 published manifests keep their verdicts under the new validator (38 valid, 3 listed drafts invalid); C2 a v1 activation record still passes its own companion calls |

Regression, run as existing tests and not edited: `tests/unit/governance/test_iiac_negative_guards.py` (N-01..N-xx), `test_iiac_activation_gate.py`, `test_iiac_g0_inventory.py`, `test_iiac_new_verifiers.py`; `verify_iiac_policy_activation.py` dry run (non-certifying); the 3-clone `test_evol_x_real_repo_fresh_clone.py` is listed as regression scope with a stated runtime risk (not run when the plan was drafted).

## 7. Write set

The only paths the pull request may touch (one per line):

```text
scripts/governance/iiac_common.py
scripts/governance/verify_iiac_audit_manifest.py
scripts/governance/verify_iiac_role_separation.py
scripts/governance/iiac_activation_gate.py
docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md
tests/unit/governance/test_e05_role_separation_identity.py
governance/tickets/T-IGX-044-F.md
governance/tickets/plans/PLAN_T-IGX-044-F.md
```

Product paths: none (`scripts/`, `tests/` and `docs/` are not under `intergrax/`, `agents/`, `applications/`). Five paths are material governance paths; one is new; two are the ticket and its plan. If an existing test must change because a verdict moved, that is an amendment of this plan, not a silent edit.

## 8. Test contract

One process at a time, under the shared pytest lock `D:/Projekty/_wt/_locks/pytest.lock`, at most 15 minutes:

- One `uv run pytest tests/unit/governance/test_e05_role_separation_identity.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_g0_inventory.py tests/unit/governance/test_iiac_new_verifiers.py -o cache_dir=<scratch dir>`.
- One `ruff check` and one `ruff format --check` over the changed Python files.
- Red-first: the new file run once on a detached worktree at the pre-change commit (the 27 negative tests must fail there); the log is kept for the auditor.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the paths of section 7.
- No broad or diagnostic suite, no Docker, no vendor integration (C10/C15 freeze).

## 9. Independent audit, approval route, risks and rollback

- Independent audit AUD-ENF-2 at the exact head SHA by the IIAC re-audit session (not the implementer, not the certifier); suggested mutation checks: revert each v2 gate (v2 marker ignored), drop the normalisation, drop `operator_line_kind`, remove one wired file from the material set.
- Approval route: this ticket changes material governance paths and the meaning of the gate. Its approval line is written only through the wave packet that the Operator approves (OD-AF-22) or by the Operator personally, never by the signing delegate on its own authority.
- Risk: after merge the activation gate reports ACT-N11 stale audit until the RB-2 re-pin (section 3); this is intended and recorded, not suppressed.
- Risk: `iiac_common.py` is imported by the PT-E04 verifiers and named by `lint_iiac_closure_regex.py`, `verify_od_ent_02_compensating_control.py` and two qualification preflights (`git grep` over four IIAC script names; which of them import it is UNVERIFIED); the regression run covers their tests, the preflights are not run.
- Risk: v2 detection by a new field means a v2 manifest can be made to look v1 by omitting the marker; mitigation: a manifest committed after the cut-over SHA recorded in the protocol document must carry the marker (open question Q-E05-01 for the cut-over rule).
- Rollback: revert the single pull request; the activation record file is not touched.

## 10. Closure condition

The pull request is merged after review and the AUD-ENF-2 audit, and the ticket records the merge SHA. No finding is marked closed or verified by this ticket; A2-038 stays open in the register until the RB-2 re-audit has run on the integrated head.
