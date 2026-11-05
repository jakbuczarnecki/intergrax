# IIAC Session B — Independent governance auditor instructions

**Precondition:** Session A complete @ pinned `GOVERNANCE_IMPLEMENTATION_SHA`.

**Role:** `independent-quality-auditor` — **new chat** — `@intergrax-independent-audit`

## Input

- Full diff @ `GOVERNANCE_IMPLEMENTATION_SHA` touching `docs/project/maintainers/qualification/**`, `scripts/governance/verify_iiac_*`, `governance/tickets/T-IGX-016-A.md`
- [`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`](./INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md)

## Execute

1. Adversarial review — no product/governance fixes in audited scope.
2. Run:
   - `uv run pytest tests/unit/governance/test_iiac_*.py -q`
   - `uv run python scripts/governance/verify_iiac_audit_manifest.py <manifest>`
   - `uv run python scripts/governance/verify_iiac_role_separation.py <manifest> --strict-product-diff`
   - `uv run python scripts/governance/verify_od_ent_02_compensating_control.py`
3. Mode C spot checks on G0 matrix samples (category 1).
4. Publish manifest `stage_id=IIAC-GOV-1`, verdict PASS|FAIL, `audit_report_digest`.

## Forbidden

- Setting `IIAC_POLICY_ACTIVATION_RECORD.json` to `IN_FORCE` without operator line (certifier/operator step after PASS).
- Editing implementer commits in audited scope.

## Output

- Findings list + scoped verdict
- Handoff to certifier/operator for activation (§15 master plan)
