# T-IGX-019-A — EVOL-X T-W4-01 consolidated corrective

## Scope

One consolidated corrective for the blocking findings of the T-W4-01 independent delta re-audit
(`PARTIAL / EXACT BLOCKERS`, audited implementation `6cf8c5a22`, hotfix `1e28b9e2b`, baseline
`5afaad9eadfc1655f4ef1ab59eb022960a924648`). Source: auditor package `EVOL_X_W4_01_CORRECTIVE_SPECIFICATION.md`,
`EVOL_X_W4_01_INDEPENDENT_AUDIT_FINDINGS.json` (held outside the repository; not edited).

Part 1 — implementer-only (this ticket's delivered scope):

- **RA-B4-05** exact certified-scope enforcement against the P2 blob (`certification_publication_sha`):
  limitations as an `(id, text)` multiset, `certification_verdict`, every FRZ-EVOL disposition, scope/claim keys of
  the manifest and certification record (`A-19`); promoted FRZ rows rendered from the baseline disposition, B6 record
  line allow-list, canonical title/scope and `Accepted limitations` = baseline `(id, sha256(text))` set (`B-11`).
  No amendment channel: any deviation needs a new P2 certification publication.
- **W4-F03** lifecycle for blockers added after the baseline (walk of the certification-record history);
  explicit error for duplicate open ids.
- **W4-F04** reject operator/P3/OD signatures older than the certification publication and negated scope text.
- **W4-F06** `-text` for the evidence-publication binding, its UTF-8 republication and the delegation record;
  the original binding is not edited — a new evidence-only commit adds `…BINDING.utf8.json`.
- **W4-F07** tests that run the real activation gate and the real preflight in fresh `core.autocrlf=true|false`
  clones of the repository.

Part 2 — BLOCKED until an Operator decision exists (material governance file), NOT done here:

- **W4-F01** make `IIAC_POLICY_ACTIVATION_RECORD.json` `handoff_digest` checkout-independent (option A: `-text` on the
  handoff record and re-bind the digest to the blob; option B: ACT-N08 digests the committed blob). This ticket does
  not touch `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_*`, the protocol or the activation record. The
  `autocrlf=false` activation checks are strict `xfail` (they flip red when W4-F01 is repaired).

Files: `scripts/qualification/evol_x_closure_publication_preflight.py`,
`tests/unit/qualification/test_evol_x_closure_preflight_b4.py`, `tests/unit/governance/test_evol_x_real_repo_fresh_clone.py`,
`.gitattributes`, `docs/project/maintainers/qualification/IIAC_EVOLX_B4_EVIDENCE_PUBLICATION_BINDING.utf8.json`
(new, evidence-only commit), `docs/project/maintainers/qualification/evol_x/EVOL_X_T_W4_01_CORRECTIVE_REAUDIT_HANDOFF.md`.
No `intergrax/`, `agents/`, `applications/`.

## Plan digest

Source: T-W4-01 auditor dossier @ `5afaad9eadfc1655f4ef1ab59eb022960a924648`

## Approved

Approved: ________ (operator signature required — not signed by the implementer or the auditor)

## Test contract

```powershell
uv run pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py tests/unit/governance -q --basetemp=D:/w4pt/x
# fresh clones, autocrlf=true AND false (see tests/unit/governance/test_evol_x_real_repo_fresh_clone.py)
# real-evidence widening probes (auditor evidence/real_scope_probe*.py): every widened variant must now add an A-19 error
# GitHub Actions governance-fast on the pushed SHA: only the W4-F01 tests may remain red until Part 2
```

## Out of scope

B6 acceptance, FRZ promotion, OD-EVOL-12 / OD-EVOLX-08 / P3 decisions, recording B-4 as resolved, EVOL-X closure,
any edit to existing evidence, W4-F01 (Part 2), W4-F08 (operator attestations), the activation gate and IIAC protocol.

## Lifecycle state

draft

## Checkpoint

Part 1 implemented by the implementer role; the Approved line is left for the operator. The implementer cannot
certify its own correction: a new `independent-quality-auditor` session must re-audit (delta), then a separate
certifier accepts. See `EVOL_X_T_W4_01_CORRECTIVE_REAUDIT_HANDOFF.md`.
