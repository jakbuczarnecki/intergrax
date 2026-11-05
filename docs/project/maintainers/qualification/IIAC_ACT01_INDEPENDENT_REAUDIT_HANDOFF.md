# IIAC ACT-01 — handoff for a new independent auditor

**Role required:** `independent-quality-auditor`  
**Must be a new session.** The implementing session does not issue PASS.  
**Policy:** `IIAC-2026-11-05.1`  
**Parent baseline (pre-ACT-01):** `5d6b44b05f90ecf420d45d01659ea64565b52848`  
**Historical governance SHA:** `2ae1014116d353bc7742985e62f2bccc2bfac798`  
**Re-audit report digest (unchanged bytes):** `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde`  
**ACT01_CORRECTIVE_IMPLEMENTATION_SHA:** the commit whose parent is the baseline above and whose tree contains this handoff. Do not treat a SHA written inside that same commit as self-proof. Read `git rev-parse HEAD` after fetch.

## Original finding

Certifier ACT-01: the activation gate accepted `IN_FORCE` with digest `deadbeefdeadbeef`, a missing manifest path, verdict `FAIL`, and a null operator line, and it returned exit 0 for live `NOT_IN_FORCE` without a flag.

## Root cause

Success depended on a nonempty digest string, a nonempty path string, and a 40-character ancestor SHA. Artifact bytes, verdict, operator decision, and companion verifiers were not consulted. Inactive status was treated as success.

## Files

- `scripts/governance/verify_iiac_policy_activation.py`
- `scripts/governance/iiac_activation_gate.py`
- `scripts/governance/iiac_common.py` (`repo` argument on git helpers)
- `scripts/governance/verify_iiac_audit_manifest.py` (`--repo`)
- `scripts/governance/verify_iiac_role_separation.py` (`--repo`)
- `tests/unit/governance/test_iiac_activation_gate.py`
- `docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/` (byte copies)
- Protocol, migration matrix, ECAP IIAC row, publication contract, activation notes
- `IIAC_ACT01_CORRECTIVE_SPEC.md`, this handoff, the dossier, the unsigned operator packet
- `governance/tickets/T-IGX-016-B.md`

## Changed behavior

- Default and `--require-in-force`: exit 0 only when every activation proof passes.
- `--expect-not-in-force`: exit 0 only for `NOT_IN_FORCE` / `DRAFT` / `PENDING` without an active claim.
- Verdict must be `PASS` on `stage_id=IIAC-GOV-1`. `FAIL`, `CONDITIONAL`, `PARTIAL`, and `PASS_SCOPED` do not activate.
- SHA-256 of report, manifest, scope, matrix, findings, and handoff must match.
- Paths must stay inside authorized roots. `..`, absolute paths, and symlinks are rejected. `historical_session_b/` cannot be the active manifest.
- Operator file must carry decision id, policy version, scope `IIAC policy activation`, `Decision status: AUTHORIZED`, the audit digest, and the exact activation line. `Approved` and the Session A line do not qualify.
- `effective_from_sha` is the phase-1 commit (blob already `IN_FORCE`, effective SHA empty). It must not equal HEAD.
- The gate executes `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py --strict-product-diff`.
- Stage-closure fields or nearby `CLOSED` claims for EVOL-X, ENT-AUDIT-X, or ARCH-FREEZE fail closed.

## Qualification already run by the implementer (not an audit)

45 passed, 1 skipped (`test_act_n14_symlink_escape`: symlink creation unavailable). ACT-N14 traversal returned `ACT-N14`. Canonical default / `--require-in-force` returned `ACT-N01`. `--expect-not-in-force` returned 0. G0 completeness PASS at `afe9e8ab`. Closure preflights stayed exit 1.

## Commands to replay

```powershell
uv run pytest tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_g0_inventory.py -q
uv run python scripts/governance/verify_iiac_policy_activation.py
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
uv run python scripts/governance/verify_iiac_policy_activation.py --expect-not-in-force
uv run python scripts/governance/verify_iiac_g0_inventory_completeness.py
```

Confirm report digests:

- reaudit `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde`
- historical `56c9b078e42e1408799e37a2535aa61879e31021864214e46dfa165fb41fded1`

## Evidence publication

Durable copies live in `docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/`. `.tmp/session/` is not activation authority. A future activation manifest must be a certifier publication: the preserved reaudit manifest still stores `certification_publication_sha` equal to the replay SHA `5d6b44b05f90ecf420d45d01659ea64565b52848`, so the new gate will reject it as the active acceptance record until a distinct certification commit exists. Do not rewrite those bytes.

## Threat model

An activator can otherwise point `IN_FORCE` at an arbitrary digest, a missing file, a `FAIL` or `CONDITIONAL` verdict, a Session A approval string, a pre-activation commit, a self-SHA, a stale tree, an open blocker, a traversed path, or a closure claim for EVOL-X / ENT-AUDIT-X / ARCH-FREEZE.

## Residual risks

- Symlink rejection is implemented and not executed on this host.
- `--evidence-root` exists for fixtures and is refused for the canonical record only.
- Published reaudit `certification_publication_sha` remains the replay SHA on purpose.
- F03 operator handoff line is still `PENDING`.
- INFO findings are not rewritten; BLOCKER/CRITICAL must be `RESOLVED` or `CLOSED`.
- Closure preflights still fail on legacy B6 gaps. That is fail-closed, not a new waiver.

## Acceptance criteria

- Replay the negative matrix; each case exits nonzero with its `ACT-N*` marker.
- Synthetic positive fixture exits 0 and does not change canonical status.
- Canonical status remains `NOT_IN_FORCE`.
- Historical and reaudit digests match the published bytes.
- No PASS record and no signed activation file were added by the implementer.
