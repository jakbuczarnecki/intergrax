# T-IGX-019-A — EVOL-X T-W4-01 consolidated corrective (one pass)

## Scope

One consolidated corrective for the blocking findings of the EVOL-X B4 independent audits (W4-01 on baseline
`5afaad9eadfc1655f4ef1ab59eb022960a924648`; delta audit on `81418b0dd`): `PARTIAL / EXACT BLOCKERS`. Source package
`EVOL_X_W4_01_CORRECTIVE_SPECIFICATION.md` / `EVOL_X_W4_01_INDEPENDENT_AUDIT_FINDINGS.json` (held outside the repository, not edited).
The delta-audit dossier (`EVOL_X_B4_DELTA_REAUDIT_DOSSIER.md`) was not found in the local workspace: its findings (DA-01, DA-02, DA-09)
were taken from the operator mission text and reproduced independently by tests before being closed.

- **DA-01** trusted P2 anchor: `P2_CERTIFICATION_PUBLICATION_SHA` is a module constant; no field of the evidence under test is read as the
  baseline (including fixture/`--repo` runs: no fallback to the declared `certification_publication_sha`); the anchor must be the commit that
  first published the certification record and a strict ancestor of HEAD; manifest and record must carry it.
- **DA-02 / RA-B4-05** exact certified scope: limitation multiset, verdict, FRZ dispositions, scope/claim keys, manifest bytes (M0), now also every
  non-lifecycle key of the certification record (`stage_id`, `policy_version`, `independent_audit_verdict`, `notes`, ... sealed), dropped keys,
  and byte-identical P2 evidence files (assessment, handoff, validator runs) bound by the P2 index.
- **W4-F03/F05** blocker lifecycle for blockers added after P2; explicit duplicate-id errors.
- **W4-F04** operator records: signature older than P2, negation, homoglyph/zero-width/non-ASCII scope text, revocation-type fields.
- **W4-F06 / DA-09** `-text` for the binding records; append-only UTF-8 republication `...BINDING.utf8.json` (original untouched).
- **W4-F07** real-repository fresh-clone tests (`autocrlf` true / false / input).
- **W4-F01** non-material repair: `text eol=crlf` for the activation handoff record in `.gitattributes` (digest meaning, blob, activation
  record and gate unchanged; no `MATERIAL_GOVERNANCE_PATHS` entry touched).
- CI: new `evol-x-b4-preflight` job; `governance-fast` timeout 8 -> 15 min (three real-repo clones added).

Not done here (authority boundary): **W4-F08** operator attestation of historical session identities; operator decisions OD-EVOL-12 /
OD-EVOLX-08 / P3; B6 acceptance; FRZ promotion; EVOL-X closure; any edit to existing evidence, the activation record, the activation gate or
the IIAC protocol.

Files: `scripts/qualification/evol_x_closure_publication_preflight.py`, `tests/unit/qualification/test_evol_x_closure_preflight_b4.py`,
`tests/unit/governance/test_evol_x_real_repo_fresh_clone.py`, `.gitattributes`, `.github/workflows/unit-tests.yml`,
`docs/project/maintainers/qualification/IIAC_EVOLX_B4_EVIDENCE_PUBLICATION_BINDING.utf8.json` (new, evidence-only),
`docs/project/maintainers/qualification/evol_x/EVOL_X_W4_01_CONSOLIDATED_CORRECTIVE_IMPLEMENTATION_REPORT.md` and the re-audit handoff.
No `intergrax/`, `agents/`, `applications/`.

## Plan digest

Source: operator mission "INTEGRAX - EVOL-X B4 - ONE-PASS CONSOLIDATED CORRECTIVE IMPLEMENTATION" @ `5afaad9eadfc1655f4ef1ab59eb022960a924648`

## Approved

Approved: operator @ 2026-11-06 (mission "ONE-PASS CONSOLIDATED CORRECTIVE IMPLEMENTATION" sections 14-15: create the consolidated corrective ticket and implement; the mission does not authorise B6, FRZ promotion, closure or material governance edits)

## Test contract

```powershell
uv sync --extra dev-ci --frozen
uv run pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py -q -n auto
uv run pytest tests/unit/governance -q
uv run python scripts/qualification/evol_x_closure_publication_preflight.py --gate b6-readiness   # expected: only the real A-18 blockers
git diff 5afaad9ea..HEAD --stat   # no intergrax/ agents/ applications/ activation record/gate/protocol
```

## Out of scope

B6 acceptance, FRZ promotion, OD-EVOL-12 / OD-EVOLX-08 / P3 decisions, recording B-4 as resolved, EVOL-X closure, any edit to existing
evidence, the IIAC activation record / gate / protocol, W4-F08 attestations.

## Lifecycle state

ready_for_audit

## Checkpoint

Implemented by an implementer session; it cannot certify its own correction. A new `independent-quality-auditor` session must re-audit the exact
implementation SHA named in the handoff; a separate certifier then accepts. See `EVOL_X_W4_01_CONSOLIDATED_CORRECTIVE_IMPLEMENTATION_REPORT.md`.
