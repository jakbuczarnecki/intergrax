# EVOL-X W4-01 consolidated corrective — independent re-audit handoff

**For:** a NEW `independent-quality-auditor` session (not `local_4b071fe5-82d0-48bc-b524-146a3daf5c45`, not any session listed in §H of the report).
**Ticket:** `T-IGX-019-A` · **Policy:** `IIAC-2026-11-05.1` · **Audit baseline:** `5afaad9eadfc1655f4ef1ab59eb022960a924648`
**Pinned SHA to audit:** `9f8b04d2b8b670077b1e78ce812bb34b680f3968` (published: `origin/development` @ `49b50fd74`, which differs from the pinned SHA by documentation and ticket text only; report §M) · the report and this handoff are in a later documentation commit; audit the code at the pinned SHA.
This handoff is evidence for an audit. It is not a PASS record, resolves nothing and authorises nothing. EVOL-X is `NOT_CLOSED`.

## 1. What to verify

1. `git diff 5afaad9ea..9f8b04d2b8b670077b1e78ce812bb34b680f3968 --stat`: no `intergrax/`, `agents/`, `applications/`, `scripts/governance/*`, activation record, certification protocol or existing evidence; the original binding and delegation record byte-identical to the baseline.
2. DA-01: `P2_CERTIFICATION_PUBLICATION_SHA` equals the first commit adding `EVOL_X_IIAC_CERTIFICATION_RECORD.json`; with `--repo <clone>` and no `--p2-sha` the pin is used; re-pointing any field cannot move the baseline.
3. DA-02: try to widen scope through a path **not** listed in the report (e.g. lifecycle keys, `certification_publication_sha_binding`, new evidence files in the certification directory, B6 record content) and try a legitimate operation that the new equality wrongly rejects.
4. W4-F01: confirm that `text eol=crlf` is an acceptable, non-material repair and that no `MATERIAL_GOVERNANCE_PATHS` file changed.
5. Challenge the residuals in the report (§C limitations, §J OD-1).

## 2. Commands (Windows `core.autocrlf=true`; use a short `--basetemp`)

```powershell
git checkout 9f8b04d2b8b670077b1e78ce812bb34b680f3968
uv sync --extra dev-ci --frozen
uv run pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py -q -n 6 --basetemp=D:/w4a/bt   # 423 tests expected green (serial ~12+ min)
uv run pytest tests/unit/governance -q -n 4 --basetemp=D:/w4a/bg                                                         # includes real-repository fresh clones (autocrlf true/false/input)
uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force                                      # IN_FORCE
uv run python scripts/qualification/evol_x_closure_publication_preflight.py --gate b6-readiness                           # expected FAIL with exactly two A-18 open blockers (B-4, B6-P3); no A-19
uv run python scripts/qualification/evol_x_closure_publication_preflight.py --gate formal-closure                         # expected FAIL (B6, FRZ, operator decisions not done)
```

Linux: clone the bundle/branch inside a `python:3.12` container (no `core.autocrlf`) and run the same commands; the activation gate must pass without any change.

## 3. Expected outcomes of the adversarial fixtures

* Every case in the report §F is a **rejection on both gates**; the only positive controls are `…positive_control…`, `…reordering_only_is_a_positive_control[*]` and `test_w4_f04_valid_record_is_the_positive_control`.
* The canonical repository must **not** reach PASS on Gate A or Gate B while B-4 / B6-P3 / B-5 stay open.

## 4. Known limits (do not count as new findings unless you can exploit them)

* The pin is a constant in a non-material script (OD-1). History rewrite is outside the model. The operator signature line is process-enforced text.
* Raw auditor evidence of the delta audit (DA-01/DA-02/DA-09) was not available locally; the findings were reproduced from the mission description.
* GitHub Actions run 42210092666 (head 445d5883c): governance-fast 146 passed, evol-x-b4-preflight 423 passed (report §L). `ent-audit-fast` fails on `development` too and is unrelated.

## 5. Session identity (W4-F08)

Implementer: `local_4b071fe5-82d0-48bc-b524-146a3daf5c45` (verified). The historical implementer ids are `UNVERIFIED / OPERATOR_ATTESTATION_REQUIRED`. Fill your own session id in the auditor manifest; the certifier must be a third session.

## Round 2 addendum (after the W401 re-audit of `9f8b04d2b`: PARTIAL)

Audit the code at **`ca78b1c1f`** (commits after it are documentation only; verify with `git diff ca78b1c1f origin/development -- . ':!docs' ':!governance/tickets'`). A new independent-quality-auditor session is preferable; a different session than the implementer and the certifier is mandatory. Check specifically: the six W401 fixes in report §N (replace refs / grafts / shallow, lifecycle values, certification-directory freeze, duplicate keys, HEAD-bound inputs, `--full-history`); that no legitimate operation is now rejected (a normal clean Windows checkout with `autocrlf=true`, the real chain); and the residual findings RA-05, RA-07, RA-08 left open on purpose. The `w401` tests are the reproductions of the previous auditor's attacks.

## Round 3 addendum (after the W402 delta re-audit of `ca78b1c1f`: PARTIAL, blocker W402-01)

Audit the code at **`2e0ed9c6b`** (commits after it are documentation only; verify with `git diff 2e0ed9c6b origin/development -- . ':!docs' ':!governance/tickets'`). Re-run your own W402 repros (`skip-worktree`, `assume-unchanged`, same-size edit, verifier-code edit, free-text channels, forged commit-graph, `core.fsmonitor`) and variants; check that no legitimate operation is now rejected (clean checkouts with `core.autocrlf` true/false/input, a linked worktree, an unrelated dirty file, the real chain: canonical `b6-readiness` must still fail with exactly the two `A-18` blockers). Residuals left open on purpose: W401-RA-05, RA-07 (D9), RA-08 (OD-1, decided C, re-opened before P3), and the unverified items listed in report section O.
