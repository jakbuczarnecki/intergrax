# Echo agent - architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

## Purpose

Minimal **Harness reference agent** for gate tests, lab roster, and `poc_template_application`.

## Capabilities

- `echo.basic`

## Skills

- `harness.tool_smoke` (via contract)

## Runtime

- `HarnessReferenceAgent` + single UAEP pipeline step
- Optional `LabHarnessContext` injected by `lab_application` host builders
- Imports only `intergrax.*` and `agents/echo` (no Tier-3 `applications` imports)

## Registration

- `applications/lab_application/manifest.py` when `LAB_INCLUDE_ECHO=true`
- `applications/poc_template_application/manifest.py` (default)
