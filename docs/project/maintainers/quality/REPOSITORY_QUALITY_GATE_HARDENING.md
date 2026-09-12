# Repository quality gate (HARDENING-9)

## Problem

`pytest tests/unit --collect-only` reported three collection errors. Imports of MP-4B symbols from `intergrax.contracts.decision` failed while integration tests under `intergrax.contracts.decision.integration` still worked.

## Cause

The package directory `intergrax/contracts/decision/` shadows the sibling module file `intergrax/contracts/decision.py`. Python resolves `intergrax.contracts.decision` to the package `__init__.py`, which previously exported only integration adapters—not `DecisionId`, `SCHEMA_DECISION_OUTCOME_V1`, or related MP-4B contracts used by `intergrax/contracts/approval.py` and unit tests.

## Fix

Re-export the MP-4B public surface from `decision.py` via `importlib` in `intergrax/contracts/decision/__init__.py`, alongside existing integration exports. No change to MP-4B contract semantics.

Added architecture gate `tests/unit/runtime/architecture/test_repository_quality_gate.py` to guard collection and the decision namespace import.

## Current gate

```bash
uv run pytest tests/unit --collect-only -q
# Expect: 0 collection errors

uv run pytest tests/unit
# Enterprise unit suite (full run; duration depends on environment)
```
