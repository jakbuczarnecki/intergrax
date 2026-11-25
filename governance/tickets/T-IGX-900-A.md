# T-IGX-900-A - FACTORY-E2E: throwaway demo for the Cursor software factory

## Scope

Throwaway demo plan for the headless E2E proof of T-IGX-057-A. Lives only on branch `scratch/factory-e2e`, which
is never merged or pushed. Adds a tiny text helper, reader, writer and CLI under `scripts/demo/`.

## Approved

Approved: operator @ 2026-11-22T12:00:00+01:00

Approval source: covered by the Operator's acceptance of `agent-env/plans/cursor-software-factory.md`
(Verification step 3: demo plan on a throwaway scratch branch).

## Write set

```text
scripts/demo/**
scripts/demob/**
tests/unit/demo/**
tests/unit/demob/**
```

## Test contract

```text
uv run pytest tests/unit/demo -q
```

## Lifecycle state

in_progress
