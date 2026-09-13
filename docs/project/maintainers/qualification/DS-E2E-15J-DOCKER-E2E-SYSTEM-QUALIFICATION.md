# DS-E2E-15J — Docker E2E System Qualification

**Task:** `DS-E2E-15J-DOCKER-E2E-SYSTEM-QUALIFICATION`  
**Type:** Docker-hosted integrated qualification (harness + tests; no new runtime capability)  
**Branch:** `development`

**Canon:** [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) · [`DECISION_SYSTEM_ARCHITECTURE.md`](../../architecture/DECISION_SYSTEM_ARCHITECTURE.md) · [`EXECUTION_ENGINE.md`](../architecture/EXECUTION_ENGINE.md)

---

## Docker E2E Qualification Summary

**Status:** **QUALIFIED**

All nine in-container scenarios pass via `tests/integration/decision_system/test_docker_e2e_system_qualification.py` (25 tests including parametrized gates; Docker daemon required). Local parity: `tests/unit/testing_support/decision_e2e/test_docker_system_scenarios.py`. Image: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` (same pattern as DS-E2E-06). Harness uses an isolated in-container `UV_PROJECT_ENVIRONMENT` so bind-mounted Windows workspace `.venv` is never mutated; result paths use POSIX `/durable/...` strings (not host `Path`).

---

## Environment Validation Matrix

| Obszar | Status | Uwagi |
| ------ | ------ | ----- |
| Container startup | PASS | `startup-health` via `docker_system_worker` |
| Configuration | PASS | Single composition root; no docker-only business branch |
| Decision flow | PASS | L6 orchestration in container |
| Governance | PASS | ALLOW / BLOCK / REQUIRE_APPROVAL |
| Execution Engine | PASS | Recording execution provider; integration boundary unchanged |
| Evidence | PASS | Correlation IDs + audit envelope |
| Failure handling | PASS | Execution fault, missing governance, admission deny |
| Plugin loading | PASS | Custom lifecycle adapter in container |
| Security | PASS | No execution-runtime import in decision composition module |

---

## Proof tests

| Test | Path |
| ---- | ---- |
| Local scenario parity (9 scenarios) | `tests/unit/testing_support/decision_e2e/test_docker_system_scenarios.py` |
| Docker in-container (requires daemon) | `tests/integration/decision_system/test_docker_e2e_system_qualification.py` |

Harness: `testing_support/decision_e2e/docker_system_scenarios.py`, `docker_system_worker.py`, `docker_system_qualification.py`.

---

## Environmental constraints

- Requires Docker CLI + running daemon for integration markers (`pytest.mark.docker`, `no_ci`).
- Mounts repository at `/workspace` and durable artifacts under `.tmp/decision_e2e_qualification/`.
- In-container `uv sync` targets `/opt/intergrax-decision-e2e-system-qual-venv` (`UV_LINK_MODE=copy`); do not rely on mutating the host `.venv` through the bind mount (especially on Windows).
- Does **not** add containers, services, or alternate execution paths beyond the existing DS-E2E worker image pattern.

---

## System diagram (qualified flow)

```text
                    Decision System
                           ↓
                 Governance Authorization
                           ↓
                    Execution Engine
                           ↓
                       Runtime
                           ↓
                  Evidence / Audit Layer
```
