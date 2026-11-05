# IIAC ACT-01 — independent re-audit dossier

**Role:** `independent-quality-auditor`  
**Policy:** `IIAC-2026-11-05.1`  
**Branch:** `development` (`refs/heads/development`)  
**Audited SHA:** `ae467c8c4dd9c87521c1c60a7e5e75224d3214a0`  
**Replay SHA:** `ae467c8c4dd9c87521c1c60a7e5e75224d3214a0` (clean worktree)  
**Canonical status observed:** `NOT_IN_FORCE`  
**This dossier is an audit record.** It does not set `IN_FORCE`, does not sign an operator decision, and does not certify the policy.

## Verdict

`PASS — ACT-01 INDEPENDENTLY VERIFIED`

No exact blocker. The fail-closed gate, the ACT-N01..ACT-N18 matrix (symlink case excepted and substituted), the byte restoration `2a6545f1`, and the live `NOT_IN_FORCE` record hold at the pinned HEAD.

## Ancestry

| SHA | Parent | Subject |
|---|---|---|
| `5d6b44b05f90ecf420d45d01659ea64565b52848` | — | parent baseline (pre-ACT-01) |
| `1d0722723687fb93d0a538161ca4cce376b3ecef` | `5d6b44b0` | Fix IIAC activation gate so IN_FORCE requires bound evidence |
| `2a6545f1d551814a0ba8896be79576c580353771` | `1d072272` | Preserve original IIAC audit evidence bytes in git |
| `ae467c8c4dd9c87521c1c60a7e5e75224d3214a0` | `2a6545f1` | Record ACT-01 gate and evidence-preservation SHAs |

`git merge-base --is-ancestor` is true for `ae467c8c`, `2a6545f1`, `1d072272`, `5d6b44b0`, and historical governance `2ae1014116d353bc7742985e62f2bccc2bfac798`.  
`ae467c8c` changes only the corrective dossier, the re-audit handoff, the unsigned operator packet, and `governance/tickets/T-IGX-016-B.md`. Gate sources and `iiac_evidence/**` bytes are identical to `2a6545f1`.  
Product diff `5d6b44b0..ae467c8c` for `intergrax/`, `agents/`, `applications/`: empty.

## Executed tests

```text
uv run pytest tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_g0_inventory.py -q --tb=line
45 passed, 1 skipped in 114.54s
PYTEST_EXIT=0
SKIPPED test_act_n14_symlink_escape: symlink creation is unavailable
```

The skipped test is the file-symlink case. `test_act_n14_path_traversal` ran inside the same suite and passed. The positive fixture passed and left the canonical record `NOT_IN_FORCE`.

| Command | Exit | Observed result |
|---|---|---|
| `verify_iiac_policy_activation.py` | 1 | `ACT-N01 status NOT_IN_FORCE is not activation eligibility` |
| `verify_iiac_policy_activation.py --require-in-force` | 1 | `ACT-N01 status NOT_IN_FORCE is not activation eligibility` |
| `verify_iiac_policy_activation.py --expect-not-in-force` | 0 | `IIAC policy activation: NOT_IN_FORCE` |
| `verify_iiac_g0_inventory_completeness.py` | 0 | PASS, 17510 paths @ `afe9e8abeabdbab410bf562ebed6f4a309a91057` |
| `verify_od_ent_02_compensating_control.py` | 0 | PASS (package draft) |
| `verify_iiac_audit_manifest.py` on reaudit manifest | 0 | PASS |
| `verify_iiac_role_separation.py --strict-product-diff` on reaudit manifest | 0 | PASS |
| same two companions on `historical_session_b` manifest | 0 | schema and role-separation PASS; verdict in the file remains `CONDITIONAL` |
| `evol_x_closure_publication_preflight` | 1 | missing signed EVOL B6 record; OD-EVOL rows still PENDING |
| `ent_audit_x_formal_closure_preflight` | 1 | missing signed ENT B6 record; `formal_b6_status` is not ACCEPTED; FRZ-AUD-01 still OPEN |

ACT-N01..ACT-N18 markers are asserted by `tests/unit/governance/test_iiac_activation_gate.py`. ACT-N12 is covered by the pre-activation blob case and the self-HEAD case. Both passed.

## Evidence digest validation

`EVIDENCE_INDEX.json` lists 10 files, `policy_status=NOT_IN_FORCE`. Worktree bytes, `2a6545f1` blobs, and `ae467c8c` blobs match the index SHA-256 and lengths (`git -c core.autocrlf=false show`).

| Artifact | SHA-256 | Match |
|---|---|---|
| Reaudit report | `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde` | index, manifest, worktree, HEAD blob |
| Reaudit manifest | `c6c666f100573b8b3cde38c81fe40f430c75c8464f05bc98caf2b21338a7c141` | index and worktree |
| Reaudit scope | `506b7474b12fde83fb634018dbf821cbbc4ed60354208ab647c0b8a48f256d03` | manifest field and bytes |
| Reaudit matrix | `650db06d2caa8d3c893563ef3a6c20b56886754d93b4ad7d295d5425c4e0467d` | manifest field and bytes |
| Reaudit findings | `e7beab6bb1d879805fb96c88b0a58dc1e2ec37ee7611973895939ae7e4521b54` | manifest field and bytes |
| Historical report | `56c9b078e42e1408799e37a2535aa61879e31021864214e46dfa165fb41fded1` | index, manifest, worktree, HEAD blob |
| Cited session handoff `IIAC_SESSION_HANDOFF_RECORD_2026_11_05_SESSION_A_B.md` | `24117bdc50bf291e534a4d144637ae226bfbf29034192fe2f814866495c721bd` (2339 bytes) | path resolves; the preserved manifest publishes no `handoff_digest` field |

`1d072272` evidence blobs are shorter and LF-only (reaudit report 12104 bytes, 0 CR LF versus restored 12297 bytes, 193 CR LF). They do not match the published digests. `2a6545f1` restores those bytes. `.gitattributes` contains `docs/project/maintainers/qualification/iiac_evidence/** -text`.

Preserved reaudit manifest: `verdict=PASS`, `audited_implementation_sha=5d6b44b0`, `certification_publication_sha=5d6b44b0`. Historical manifest: `verdict=CONDITIONAL`, `audited_implementation_sha=2ae10141`, handoff path is the template. Those bytes were not rewritten.

## Adversarial findings

No finding that defeats activation binding.

Confirmed by replay and by direct calls on the canonical tree:

- Default and `--require-in-force` reject live `NOT_IN_FORCE` with `ACT-N01`.
- `--expect-not-in-force` accepts the canonical inactive record.
- `deadbeefdeadbeef`, missing manifest, `FAIL`, `CONDITIONAL`, missing operator file, bare `Approved`, and the Session A line are rejected by the suite (`ACT-N02`..`ACT-N07`).
- Direct `_operator_errors` probe rejects: Session A text, ticket line `Approved: operator @ 2026-11-05T16:58:00+01:00 (IIAC-ACT-01 corrective implementation; not policy activation)`, bare `Approved`, and arbitrary text that merely contains `Decision status: AUTHORIZED`.
- Digest mismatch, wrong audited SHA, equal implementer and auditor ids, stale governance edit, pre-activation `effective_from_sha`, self-HEAD, missing report bytes, `..` traversal, open BLOCKER, wrong policy version, empty certifier session, and `stage_closures` of EVOL-X / ENT-AUDIT-X / ARCH-FREEZE are rejected by the suite.
- Canonical path probe rejects `..`, an absolute path, the unsigned packet used as a report, `historical_session_b` as the active manifest, the handoff template, and the packet used as an operator decision (`ACT-N14` / `ACT-N06`).
- The filled session handoff path is accepted as a path. Its operator line remains `Approved handoff: operator @ ________________`.
- No `IIAC_OD_ACTIVATION_*.md` and no `*IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` are present.
- `effective_from_sha` on the canonical record is null.
- Product paths between the parent baseline and HEAD are unchanged. Closure preflights remain exit 1 on legacy B6 gaps. G0 stays bound to `afe9e8ab` (17510 paths) and does not promote stage closure.

## Role separation

- Auditor role in this session: `independent-quality-auditor`. No commit. No edit under `intergrax/`, `agents/`, `applications/`, or qualification sources. Probe output is confined to `.tmp/session/iiac-act01-independent-reaudit/`.
- Preserved sessions stay distinct: implementer `IIAC-GOV-A-2026-11-05` / `implementer`; F01–F03 auditor `IIAC-GOV-B-2026-11-05-reaudit-f01f03` / `independent-quality-auditor`; certifier id on that manifest is the placeholder `IIAC-GOV-CERT-PENDING`.
- `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py --strict-product-diff` were executed on both published manifests and returned exit 0. The gate calls both after a manifest loads (`_companion_errors`). The synthetic positive fixture reaches exit 0 only when those companions return 0. `validate_manifest_dict` rejects equal implementation and auditor session ids (N-01). The gate also rejects equal agent ids (`ACT-N10`).

## Activation bypass assessment

The live record cannot activate. Exit 0 requires `status=IN_FORCE` plus digest binding, `PASS` on `stage_id=IIAC-GOV-1`, ancestor SHAs, a phase-1 blob that is already `IN_FORCE` with an empty `effective_from_sha`, `effective_from_sha` different from HEAD, a certification SHA different from the corrective SHA, companion PASS, closed BLOCKER/CRITICAL findings, and an `IIAC_OD_ACTIVATION_*.md` file whose line matches `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>`.

The preserved reaudit manifest is not an activation acceptance record: its audited SHA and `certification_publication_sha` are the replay SHA `5d6b44b0`. The gate rejects a certification SHA equal to the corrective SHA and a manifest audited SHA that differs from `corrective_implementation_sha`.

`--evidence-root` on the canonical record is rejected (suite test passed). Session A approval and the ACT-01 ticket approval line do not satisfy the operator predicate.

## Remaining limitations

1. File symlink creation on this host fails with WinError 1314 (`SeCreateSymbolicLinkPrivilege`). `test_act_n14_symlink_escape` did not execute. A directory junction from inside `iiac_evidence` to an outside directory was created under `.tmp` and rejected: `ACT-N14 evidence path outside authorized root`. That is the same `Path.resolve()` plus authorized-root check an outside symlink would hit. The `_has_symlink` branch itself was not executed. `resolve()` runs before `_has_symlink`, so the outside case is classified as an unauthorized root, and an in-root symlink that resolves inside the root is not separately classified as a symlink. Differing bytes still fail the digest and `git show` comparison.
2. `--expect-not-in-force` returns no error for `{status: NOT_IN_FORCE, session_b_verdict: PASS}` when `operator_activation_line` is empty. `PASS` plus any operator line is rejected. The canonical record has a null verdict and a null operator line, and default mode still returns `ACT-N01`.
3. `--evidence-root` remains available for a non-canonical `--activation-record`. It does not change the canonical JSON.
4. F03 operator handoff line is `PENDING`. The unsigned packet `IIAC_ACT01_OPERATOR_ACTIVATION_PACKET.md` stays `NOT SIGNED`.
5. Closure preflights fail on pre-existing B6 gaps. That is fail-closed behavior. This audit does not waive them and does not close EVOL-X, ENT-AUDIT-X, or ARCH-FREEZE.
6. The preserved manifest has no `handoff_digest`. The independent digest of the cited handoff file is recorded above for the certification session.

## Certification handoff

Separate certification session only. This auditor does not authorize activation.

Bind the future activation record to this replay, not to the preserved F01–F03 manifest:

| Field | Value |
|---|---|
| Policy | `IIAC-2026-11-05.1` |
| Governance implementation SHA | `2ae1014116d353bc7742985e62f2bccc2bfac798` |
| Parent baseline | `5d6b44b05f90ecf420d45d01659ea64565b52848` |
| Gate logic | `1d0722723687fb93d0a538161ca4cce376b3ecef` |
| Corrective / evidence bytes | `2a6545f1d551814a0ba8896be79576c580353771` |
| `audited_implementation_sha` / `corrective_implementation_sha` | `2a6545f1d551814a0ba8896be79576c580353771` |
| `test_execution_sha` / `audit_execution_sha` | `ae467c8c4dd9c87521c1c60a7e5e75224d3214a0` |
| This dossier | `.tmp/session/iiac-act01-independent-reaudit/IIAC_ACT01_INDEPENDENT_REAUDIT_DOSSIER.md` |
| Reaudit report digest (prior F01–F03, unchanged) | `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde` |
| Historical report digest | `56c9b078e42e1408799e37a2535aa61879e31021864214e46dfa165fb41fded1` |
| Cited handoff digest | `24117bdc50bf291e534a4d144637ae226bfbf29034192fe2f814866495c721bd` |

The certifier publishes a new manifest and a new report digest for this ACT-01 verdict. `certification_publication_sha` has to be that publication commit, distinct from `2a6545f1`, and an ancestor of the phase-1 activation commit. Phase 1 commits `IN_FORCE` with `effective_from_sha` empty. Phase 2 sets `effective_from_sha` and `policy_activation_sha` to the phase-1 commit. Commits after `ae467c8c` that touch the six material governance paths, or any `intergrax/` / `agents/` / `applications/` path, make this replay stale (`ACT-N11`).

Operator signature stays a separate `IIAC_OD_ACTIVATION_<id>.md` with decision id, policy version, scope `IIAC policy activation`, `Decision status: AUTHORIZED`, this dossier’s digest once the certifier freezes the report bytes, and the exact activation line. The Session A line and the T-IGX-016-B approval line are implementation approvals. This session does not supply the activation line.

Replay before any `IN_FORCE` publication:

```powershell
uv run pytest tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_g0_inventory.py -q
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
```

`--require-in-force` exit 0 is a certification-session result. It is not a result of this audit.
