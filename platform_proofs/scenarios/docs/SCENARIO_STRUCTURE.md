# Scenario package structure standard

Platform Proof scenarios under `platform_proofs/scenarios/<slug>/` use a consistent layout so operators, builders, and maintainers can find entry points without hunting loose files at the package root.

## Top-level layout

| Path | Purpose |
|------|---------|
| `application/` | Scenario application domain and runtime composition |
| `contracts/` | Scenario-owned contract types shared across layers |
| `dataset/` | Dataset sources, builders, and data-pack libraries (not CLI entrypoints) |
| `proof/` | Proof-owned evaluation and evidence projection |
| `scripts/` | **All operational CLI and launcher scripts** |
| `tests/` | Scenario-local test helpers (not pytest tree under `tests/unit/…`) |
| `docs/` | Scenario-local documentation beyond `SCENARIO_SPEC.md` |
| `run_proof.py` | Thin proof runner (platform scaffold) |
| `README.md` | Public scenario page (design stage) |

## `scripts/` categories

Place every operator-facing or batch script in exactly one subdirectory:

| Subdirectory | Use when the script… |
|--------------|----------------------|
| `scripts/build/` | Generates datasets, shards, manifests; validates build artifacts |
| `scripts/operator/` | Starts, resumes, stops, or recovers long-running work; one-click launchers (`.bat`/`.sh`) |
| `scripts/diagnostics/` | Inspects state, produces reports, runs qualification or integrity checks |
| `scripts/migration/` | Converts formats or upgrades stored artifacts (short-lived; remove when done) |

**Do not** leave ad-hoc `run_*.py` at the scenario root except `run_proof.py`.

## Naming

- Prefer `run_<action>.py` or `<domain>_<action>.py` over opaque names.
- Windows launchers live next to the Python module they invoke under `scripts/operator/`.
- Deprecated repo-global launchers may remain as thin wrappers until a dedicated cleanup; new work must use scenario-local paths.

## Scaffold

`scripts/proof/init_scenario_implementation.py` creates `scripts/{build,operator,diagnostics,migration}/`, plus empty `contracts/`, `dataset/`, `tests/`, and `docs/` placeholders for new implementations.

## Example (VPI)

`verified_product_identification/scripts/operator/resume_vpi_data_pack.bat` — one-click Data Pack resume; invokes `scripts.operator.run_vpi_data_pack_resume` via `python -m`.
