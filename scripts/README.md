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

## Guidelines

- Do not add new scripts directly to the top-level `scripts/` directory.
- Prefer stable category paths in CI and documentation.
- When moving scripts, update all references in workflows, docs, bootstrap prompts, and Cursor rules.
