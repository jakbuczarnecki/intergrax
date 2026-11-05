# T-IGX-017-A — EVOL-X B-4 corrective: two-gate IIAC closure preflight

## Scope

Consolidated corrective batch for EVOL-X certification blocker **B-4**
(`EVOL_X_IIAC_CERTIFICATION_ASSESSMENT.md` §4): the closure publication preflight enforced
superseded external-human gates (false rejection) and accepted IIAC evidence by filename
(false acceptance).

- Rewrite `scripts/qualification/evol_x_closure_publication_preflight.py` as a two-gate,
  fail-closed contract: Gate A (B6 acceptance readiness) and Gate B (formal EVOL-X closure
  readiness), reusing the canonical IIAC activation gate, manifest and role-separation validators.
- Retain every Session C scope limitation; reject any manifest, disposition or FRZ row that widens it.
- Narrow, config-driven Ruff exemption for byte-preserved IIAC evidence in the changed-path gates
  (`pyproject.toml` `[tool.ruff] extend-exclude`, `scripts/ci/push_gate_config.py`).
- Adversarial regression tests with isolated git fixtures.

Re-audit handoff: [`EVOL_X_B4_CORRECTIVE_REAUDIT_HANDOFF.md`](../../docs/project/maintainers/qualification/evol_x/EVOL_X_B4_CORRECTIVE_REAUDIT_HANDOFF.md).

## Plan digest

Source: operator mission "EVOL-X B-4 CORRECTIVE IMPLEMENTATION" @ `ca2b147825ce6117cc7219e63a93d2872669cb49`

## Approved

Approved: operator @ 2026-11-05 (EVOL-X B-4 corrective implementation mission; implementer role; commit and push authorized)

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
- Any change to Session C audit evidence or P2 certification evidence
- IIAC material governance paths (activation gate, `iiac_common`, manifest / role validators, protocol)
- Product paths (`intergrax/`, `agents/`, `applications/`) and the audited EVOL-X scope
- Implementation-session self-PASS

## Lifecycle state

ready_for_audit

## Checkpoint

Implementation complete; independent re-audit required. `B4_CORRECTIVE_IMPLEMENTATION_SHA` is recorded in the handoff §9 binding commit.
