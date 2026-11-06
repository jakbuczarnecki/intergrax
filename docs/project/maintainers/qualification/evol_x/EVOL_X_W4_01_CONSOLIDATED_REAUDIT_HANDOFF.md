# EVOL-X W4-01 consolidated corrective — independent re-audit handoff

**For:** a NEW `independent-quality-auditor` session (not `local_4b071fe5-82d0-48bc-b524-146a3daf5c45`, not any session listed in §H of the report).
**Ticket:** `T-IGX-019-A` · **Policy:** `IIAC-2026-11-05.1` · **Audit baseline:** `5afaad9eadfc1655f4ef1ab59eb022960a924648`
**Pinned SHA to audit:** `__IMPL_SHA__` (branch `evolx/w4-01-consolidated`, not pushed) · the report and this handoff are in a later documentation commit; audit the code at the pinned SHA.
This handoff is evidence for an audit. It is not a PASS record, resolves nothing and authorises nothing. EVOL-X is `NOT_CLOSED`.

## 1. What to verify

1. `git diff 5afaad9ea..__IMPL_SHA__ --stat`: no `intergrax/`, `agents/`, `applications/`, `scripts/governance/*`, activation record, certification protocol or existing evidence; the original binding and delegation record byte-identical to the baseline.
2. DA-01: `P2_CERTIFICATION_PUBLICATION_SHA` equals the first commit adding `EVOL_X_IIAC_CERTIFICATION_RECORD.json`; with `--repo <clone>` and no `--p2-sha` the pin is used; re-pointing any field cannot move the baseline.
3. DA-02: try to widen scope through a path **not** listed in the report (e.g. lifecycle keys, `certification_publication_sha_binding`, new evidence files in the certification directory, B6 record content) and try a legitimate operation that the new equality wrongly rejects.
4. W4-F01: confirm that `text eol=crlf` is an acceptable, non-material repair and that no `MATERIAL_GOVERNANCE_PATHS` file changed.
5. Challenge the residuals in the report (§C limitations, §J OD-1).

## 2. Commands (Windows `core.autocrlf=true`; use a short `--basetemp`)

```powershell
git checkout __IMPL_SHA__
uv sync --extra dev-ci --frozen
uv run pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py -q -n 6 --basetemp=D:/w4a/bt   # 424 tests expected green (serial ~12+ min)
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
* GitHub Actions was not run (nothing pushed); the Linux evidence is a container run.

## 5. Session identity (W4-F08)

Implementer: `local_4b071fe5-82d0-48bc-b524-146a3daf5c45` (verified). The historical implementer ids are `UNVERIFIED / OPERATOR_ATTESTATION_REQUIRED`. Fill your own session id in the auditor manifest; the certifier must be a third session.
