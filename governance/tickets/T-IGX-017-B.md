# T-IGX-017-B — EVOL-X RA-B4-01..04 consolidated corrective

## Scope

Consolidated corrective for the four blocking findings of the independent B-4 re-audit
(`docs/project/maintainers/qualification/iiac_evidence/EVOL-X/b4_reaudit/`, auditor verdict
`EVOL-X B-4 — PARTIAL / EXACT BLOCKERS`, audited implementation `23feb2bbb`, baseline `9a8fa3634`):

- **RA-B4-01** — Gate A A-14 compared working-tree bytes of the IIAC pass record, so a default Windows
  (`core.autocrlf=true`) fresh checkout failed three times. Byte contract: canonical committed bytes; `-text`
  on the digest-bound pass record; exact CRLF-expansion tolerance only on converting paths.
- **RA-B4-02** — A-18 treated a deleted blocker as closed. Blocker lifecycle `OPEN → RESOLVED_WITH_EVIDENCE`
  against the baseline published at `certification_publication_sha`; resolutions are independently re-verified.
- **RA-B4-03** — B-07 accepted `NOT APPROVED`. Exact allow-list of decision states, negation deny-list,
  status-table-only parsing, verifiable decision records for OD-EVOL-12 / OD-EVOLX-08.
- **RA-B4-04** — Gate B accepted B6 on three self-asserted flags. New control B-05: a published B6 acceptance
  record bound by digest, verified at the pinned publication commit against the independent audit.

Files: `scripts/qualification/evol_x_closure_publication_preflight.py`,
`tests/unit/qualification/test_evol_x_closure_preflight_b4.py`, `.gitattributes` (one narrow `-text` rule),
re-audit handoff `docs/project/maintainers/qualification/evol_x/EVOL_X_RA_B4_CORRECTIVE_REAUDIT_HANDOFF.md`.

## Plan digest

Source: operator mission "INTEGRAX — EVOL-X RA-B4-01..04 Consolidated Final Corrective Implementation" @ `9a8fa3634713f69b56ce87ceaf9ed319fad27dae`

## Approved

Approved: operator @ 2026-11-06 (EVOL-X RA-B4-01..04 consolidated corrective mission; implementer role; commit and push authorized)

## Test contract

```powershell
uv run pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py -q
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
uv run python scripts/governance/verify_iiac_audit_manifest.py docs/project/maintainers/qualification/iiac_evidence/EVOL-X/certification/iiac_audit_session_manifest.EVOL-X.json --repo .
uv run python scripts/governance/verify_iiac_role_separation.py docs/project/maintainers/qualification/iiac_evidence/EVOL-X/certification/iiac_audit_session_manifest.EVOL-X.json --strict-product-diff --repo .
uv run python scripts/qualification/evol_x_closure_publication_preflight.py --gate b6-readiness   # expected FAIL: A-18 open certifier blockers only
uv run python scripts/qualification/evol_x_closure_publication_preflight.py                        # expected FAIL: Gate B
```

## Out of scope

- B6 acceptance, `B6_ACCEPTANCE_PUBLICATION_SHA`, FRZ-EVOL promotion, OD-EVOL-12 / OD-EVOLX-08 decisions, EVOL-X closure
- Any change to Session C audit evidence, P2 certification evidence or the original B-4 re-audit evidence package
- Publication of the original B-4 re-audit evidence (needs its own authorised evidence-publication path)
- IIAC material governance paths (activation gate, `iiac_common`, manifest / role validators, protocol)
- Product paths (`intergrax/`, `agents/`, `applications/`) and the audited EVOL-X scope
- RA-B4-05 semantic scope-widening detection (accepted scoped limitation, recorded as residual risk)
- Implementation-session self-PASS; independent delta re-audit is required before B-4 is closed

## Lifecycle state

ready_for_audit

## Checkpoint

Implemented; independent delta re-audit required. The implementer does not certify this correction;
EVOL-X remains `NOT_CLOSED`. See the re-audit handoff for changed code, tests, evidence and residual risks.
