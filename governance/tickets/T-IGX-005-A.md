# T-IGX-005-A — GOV-06 TRUSTED_POLICY_PIN (optional)

## Scope

Wave W2 in [`DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md): optional `TRUSTED_POLICY_PIN.yaml` + advisory CI pin compare; manifest GOV-06 status update when implemented.

## Lifecycle state

closed

## Test contract

```bash
uv run pytest tests/unit/governance/test_dcp_*.py -q
uv run python -m scripts.governance.igx_gov doctor
uv run python scripts/governance/verify_trusted_policy_pin.py
```

## Checkpoint

```
[INFO] verify_trusted_policy_pin: OK
[INFO] igx_gov doctor: OK
governance-policy-pin CI job: ADVISORY (continue-on-error)
```

## Out of scope

Mandatory SERVER enforcement without operator promotion
