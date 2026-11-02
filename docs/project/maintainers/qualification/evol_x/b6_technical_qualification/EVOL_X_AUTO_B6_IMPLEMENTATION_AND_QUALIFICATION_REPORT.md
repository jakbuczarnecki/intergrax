# EVOL-X AUTO-B6 — implementation and qualification report

**Disposition:** **`EVOL-X AUTO-B6 — IMPLEMENTED / TECHNICALLY QUALIFIED / FORMAL AUTHORITY PENDING`**

**Not claimed:** independent organizational acceptance, `B6_ACCEPTANCE_PUBLICATION_SHA`, FRZ promotion, or EVOL-X CLOSED.

## Identity

| Item | Value |
| --- | --- |
| Branch | `development` |
| START HEAD | `19d619549e57320dacb15c51d66a401111865828` |
| END HEAD | recorded at publication commit (descendant of START) |
| Active TES | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` (unchanged) |
| Active FINAL | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` (unchanged) |
| Prior CI | `41337077334` — 290 passed at that TES (unchanged; not re-dispatched) |
| Profile | `AUTOMATION_FIRST_SCOPED` |
| New operator decision | **OD-EVOL-14** — direction and implementation only |
| ECAP amendment | `ECAP-P4-EVOL-SCOPE-AUTO-B6` version `2026-11-02.1-pending` |
| ECAP activation | **`PENDING_REQUIRED_APPROVALS`** |
| Normative effect | **`NOT_IN_FORCE`** |
| B6 acceptance SHA | **PENDING** |
| EVOL-X | **OPEN** |

## Authority gate

ECAP-§1.6 keeps implementation, independent audit, and operator acceptance as separate roles. ECAP-§2.B **B6** remains the controlling acceptance rule. OD-EVOL-14 records the operator's approval of the automation-first direction for EVOL-X only. It does not name a key custodian, does not contain a security-review signature, and does not constitute ECAP maintainer publication.

The amendment stays inactive until all three missing approvals exist: ECAP maintainer publication, signed security review, and independent key custody. This commit does not supply them.

OD-EVOL-12, OD-EVOLX-08, W7-03, W7-04, W7-05, and W7-06 were not executed. FRZ-EVOL-07 stays **OPEN**.

## Implementation architecture

| Piece | Role |
| --- | --- |
| `scripts/qualification/evol_x_b6_attestation.py` | Versioned attestation schema, Ed25519 verification, governance gate, fail-closed reasons |
| `scripts/qualification/evol_x_b6_independent_verification.py` | Mode A/B orchestration plus the four separate verdicts |
| `EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json` | Machine-readable activation record (`NOT_IN_FORCE`) |
| `tests/unit/qualification/test_evol_x_b6_attestation_contract.py` | Adversarial contract tests, outside the EVOL-X VL3 tree |

Verifier report fields, kept separate:

- `technical_verification`
- `authority_verification`
- `attestation_verification`
- `formal_acceptance_eligibility`

`b6_acceptance_publication_sha` is always null in this tool's output. `evol_x_closed` is always false. A technical PASS does not set formal eligibility.

`sign_attestation` refuses `PRODUCTION_INDEPENDENT` and refuses disposition `ACCEPTED_AUTOMATED_B6`. No production private key is written.

## Security trust boundary

The implementer workspace may evaluate records and may mint only `NON_PRODUCTION_TEST_ONLY` signatures with disposition `NOT_ACCEPTED`. Formal eligibility additionally requires governance activation `IN_FORCE`, every required approval flag, custody domain `INDEPENDENT_EXTERNAL`, a matching observed Mode A/B result, the pinned TES/FINAL, CI run `41337077334`, an external replay ledger, and a signature that verifies under a non-revoked key not controlled by the implementer.

A second Cursor worktree or subagent is not that boundary. Deployment status: **`AUTOMATED B6 IMPLEMENTED / FORMAL ATTESTATION DEPLOYMENT BLOCKED`**. Dossier: [EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md](../EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md).

## Mode A orchestration

The previous Mode A step checked `git show` before `uv sync` and before the P03 module was required to exist on disk, then imported it with the parent interpreter. The verifier now syncs the FINAL worktree first, waits until the evidence blob and `p03.py` are present on disk, and runs Mode A through `uv run` in that worktree. Windows `--basetemp` outside the nested worktree is unchanged (`build/pytest-b6-independent-replay`). Mode B writes JUnit to an absolute path under the orchestrator `--session-root` so pytest in the TES worktree does not drop reports under a missing relative `.tmp/...` path. Mode B writes JUnit to an absolute path under the orchestrator `--session-root` so pytest in the TES worktree does not drop reports under a missing relative `.tmp/...` path.

**AUTO-B6 final wave (2026-11-02):** orchestrator session `.tmp/session/b6-v3/` — Mode A **PASS**, Mode B **PASS** (290/290), adversarial **PASS**, `technical_verification` **PASS**; authority **PENDING_REQUIRED_APPROVALS**; formal **NOT_ELIGIBLE**.

**OD-EVOL-15 wave:** see [EVOL_X_AUTO_B6_FULL_WAVE_EXECUTION_REPORT.md](./EVOL_X_AUTO_B6_FULL_WAVE_EXECUTION_REPORT.md). Verifier: Mode A via `uv run --directory <orchestrator repo>`; Mode B absolute JUnit path + replay outcomes JSON (Windows WinError 206 fix).

## Qualification

| Check | Result |
| --- | --- |
| Attestation contract tests | **29 passed** (`tests/unit/qualification/test_evol_x_b6_attestation_contract.py`) |
| EVOL-X VL3 population | **Unchanged** (290 node ids at TES `3c4a641b…`). New tests are under `tests/unit/qualification/`, not `tests/qualification/evol_x/` |
| New TES / new CI | **Not created.** Qualified semantics of the 290-node population were not modified |
| Formal attestation | **None.** No production signature exists |
| Independent attestation evidence | **Absent** |

Adversarial cases covered by the unit tests: forged signature, unknown trust root, revoked key, wrong attestor, implementer-controlled attestor, wrong TES, wrong FINAL, wrong policy id/version, stale window, replay, altered Mode A/B, missing CI, M-09H PASS claim, FRZ promotion, mutated tool digest, incomplete findings, insufficient authority, self-acceptance, unsigned record, and test-credential formal claim.

## E4

Not executed. OD-EVOL-14 does not sign OD-EVOL-12 or OD-EVOLX-08.

## Shortest path to EVOL-X CLOSED

1. Security review signed by a reviewer who is not the implementer.
2. ECAP maintainer publication setting `ECAP-P4-EVOL-SCOPE-AUTO-B6` to `IN_FORCE` (not this commit).
3. Provision the independent attestor, trust root, replay ledger, and immutable log described in the deployment dossier and [EVOL_X_AUTO_B6_INDEPENDENT_ATTESTOR_DEPLOYMENT_PACKAGE.md](./EVOL_X_AUTO_B6_INDEPENDENT_ATTESTOR_DEPLOYMENT_PACKAGE.md).
4. That attestor produces a `PRODUCTION_INDEPENDENT` attestation at the pinned TES and FINAL. Publish the real `B6_ACCEPTANCE_PUBLICATION_SHA` only from that record.
5. Operator adjudicates FRZ-EVOL-07 Outcome A/B and signs **OD-EVOL-12** (W7-03).
6. Operator signs **OD-EVOLX-08** (W7-04).
7. Publish roadmap, checklist, and ladder closure (W7-05) at `FORMAL_CLOSURE_PUBLICATION_SHA`.
8. Publish the ENT-AUDIT-X handoff (W7-06).

Until step 4, B6 stays **PENDING**. Until step 7, EVOL-X stays **OPEN**.
