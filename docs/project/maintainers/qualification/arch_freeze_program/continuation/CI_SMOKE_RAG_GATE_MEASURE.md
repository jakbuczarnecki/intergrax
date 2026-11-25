# CI smoke and RAG gate — local measure

**Product SHA:** `689b277e62db18613a415e6b2e404864412112b8`  
**Worktree:** `D:/Projekty/_wt/intergrax/af-fc-ci` · branch `af-fc/ci-measure`  
**Stdout/logs:** `D:/Projekty/_wt/intergrax/_scratch/af-fc-ci/` (`ci-smoke.log`, `rag-gate.log`)

**Not run:** `governance-fast` (ACT-N11 may be red; out of scope for this measure).

## ci-smoke

**SSOT:** `.github/workflows/unit-tests.yml` job `ci-smoke` → `scripts/ci/run_regression_gate_ci.py --profile smoke`.

| Step | Command | Exit code | Verdict |
|------|---------|-----------|---------|
| Dependencies | `uv sync --extra dev-ci --frozen` | 0 | (prerequisite) |
| Gate | `uv run --frozen python scripts/ci/run_regression_gate_ci.py --profile smoke --skip-uv-sync` | 0 | **PASS** |

## rag-gate

**SSOT:** `.github/workflows/rag-guard.yml` job `rag-gate`.

| Step | Command | Exit code | Verdict |
|------|---------|-----------|---------|
| Dependencies | `uv sync --extra dev-ci --extra dev-ci-rag --frozen` | 0 | |
| Conformance | `uv run --frozen python scripts/maintenance/check_environment_conformance.py --profile ci` | 0 | |
| RAG-MAINT-01 | `uv run --frozen python scripts/maintenance/check_rag_maturity_labels.py` | 0 | |
| Pytest gate | `uv run --frozen pytest tests/unit/rag/ tests/unit/tools/providers/rag/ -m gate -q --tb=line` | 0 | **PASS** |

**Composite rag-gate exit code:** 0