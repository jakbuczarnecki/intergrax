# T-IGX-009-A — GOV-06 pin hasher git-blob SSOT (CRLF fix)

## Scope

Regression gate @ `618e234ee` failed `governance-policy-pin` / `governance-fast`: seven TRUSTED_POLICY_PIN mismatches on Linux CI while Windows working tree read passed. Root cause: pins generated from CRLF working-tree bytes; CI verifies LF git blob bytes. Normalize hasher to `HEAD:<path>` blob SHA-256; refresh pins; `eol=lf` for governance YAML.

## Approved

Approved: operator batch @ 2026-11-04T09:50:00+01:00 (FINAL GOVERNANCE CI CORRECTION prompt)

## Test contract

```bash
uv run python scripts/governance/verify_trusted_policy_pin.py
uv run pytest tests/unit/governance/ -q
uv run --frozen python scripts/ci/run_regression_gate_ci.py --profile enterprise --skip-uv-sync
```

## Lifecycle state

quality_gate

## Out of scope

FRZ / B6 attestation / OD / pin promotion to SERVER enforcement
