# Scripts

This directory contains repository tooling grouped by purpose. New scripts should be placed in the most specific category instead of the top-level `scripts/` directory.

| Directory | Purpose |
|---|---|
| `audit/` | Audit prompt generation, audit read scopes, and audit-specific checks. |
| `ci/` | CI/preflight/regression entrypoints. |
| `codemods/` | Automated source transformations and one-off migration helpers. |
| `docs/` | Documentation and index generation utilities. |
| `gates/` | Production, promotion, readiness, and release gate checks. |
| `maintenance/` | Repository hygiene, architecture constraints, catalog, wiring, and compatibility checks. |
| `release/` | Phase closeout, release-cycle, and ops-evidence helpers. |
| `setup/` | Local setup/bootstrap wrappers. |
| `public_adoption/` | Public-adoption/domain-specific helper scripts. |

## Examples

```bash
# Local environment setup (Windows)
scripts/setup/setup.bat

# CI smoke / regression preflight
uv run python scripts/ci/run_regression_gate_ci.py --profile smoke

# Run unit tests by marker
scripts/ci/test.bat gate

# Regenerate audit prompts
uv run python scripts/audit/generate_domain_audit_prompts.py

# Harness maintenance gate
uv run python scripts/maintenance/check_harness_no_getattr.py
```

## Environment conformance

Before declaring `BLOCKED_ENVIRONMENT`, run the canonical local diagnostic:

```bash
# Local Windows development (strict local provenance)
uv run --frozen python scripts/maintenance/check_environment_conformance.py

# GitHub Actions (shared CI contract)
uv run --frozen python scripts/maintenance/check_environment_conformance.py --profile ci
```

The default local profile proves Python provenance, `.venv` isolation, lock
consistency, Ruff, and baseline runtime packages. The CI profile is run
automatically after the frozen CI sync in each Python-executing workflow; it
proves the shared Python/venv/lock/isolation contract without requiring the
local uv-managed base-interpreter provenance. CI may provision Python through
`setup-uv` or runner infrastructure. Neither profile proves optional or heavy
integration dependencies, external services, GPU/CUDA, or every test suite.

## Guidelines

- Do not add new scripts directly to the top-level `scripts/` directory.
- Prefer stable category paths in CI and documentation.
- When moving scripts, update all references in workflows, docs, bootstrap prompts, and Cursor rules.
