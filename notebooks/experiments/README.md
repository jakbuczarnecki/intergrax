# Experiment notebooks (Phase D.4)

Interactive templates for the Intergrax laboratory workflow (§35):

```text
hypothesis → capability → register → Nexus → trace → evaluate → decide
```

| Notebook | Purpose |
|----------|---------|
| [`00_experiment_template.ipynb`](00_experiment_template.ipynb) | Blank §35 workflow — copy and fill in your hypothesis |
| [`01_echo_experiment.ipynb`](01_echo_experiment.ipynb) | Working Echo smoke test (deterministic, no network) |

## Prerequisites

- Repository root on `PYTHONPATH` (notebooks call `ensure_repo_root_on_path()`)
- `uv run jupyter lab` or VS Code / Cursor notebook kernel with project venv

## Quick start

1. Open `01_echo_experiment.ipynb` and run all cells.
2. Inspect output under `build/notebooks/` (trace + experiment registry).
3. Duplicate `00_experiment_template.ipynb` for a new capability experiment.

## Related

- [`docs/experiment_guide.md`](../../docs/experiment_guide.md) — CLI and HTTP workflow
- `intergrax/experiments/workflow.py` — `ExperimentSession` API used by notebooks
- `python -m intergrax.scaffold new-agent <name>` — scaffold a new agent module
