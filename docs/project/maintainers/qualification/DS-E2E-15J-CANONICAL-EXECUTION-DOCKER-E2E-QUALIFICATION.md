# DS-E2E-15J — Canonical Execution Docker E2E Qualification

**Task:** `DS-E2E-15J-CANONICAL-EXECUTION-DOCKER-E2E-QUALIFICATION`  
**Type:** Docker-hosted canonical Decision → Governance → Execution Engine proof  
**Branch:** `development`

**Canon:** [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) · [`DECISION_SYSTEM_ARCHITECTURE.md`](../../architecture/DECISION_SYSTEM_ARCHITECTURE.md) · [`EXECUTION_ENGINE.md`](../architecture/EXECUTION_ENGINE.md)

---

## Canonical Execution Qualification Summary

**Status:** **QUALIFIED**

Canonical success and negative-control scenarios pass in-container via `tests/integration/decision_system/test_docker_e2e_system_qualification.py` (`-k canonical`). Local parity: `tests/unit/testing_support/decision_e2e/test_canonical_docker_execution.py`. Harness extension: `testing_support/decision_e2e/canonical_docker_execution.py` (same Docker worker image as DS-E2E-15J system qualification).

**Distinction:** L6 matrix scenarios (`flow-success`, …) still use `RecordingExecutionProvider` for auditable reference-only execution. Canonical scenarios use `CanonicalDecisionFlowGate` → `mint_validated_execution_authorization` → `Execution` / `ExecutionRuntime` from `build_qualification_composition` (deterministic structured fake LLM adapters; no `RecordingExecutionProvider` on the success path).

---

## Canonical path used

```text
DecisionFlowRequest (QualificationRecommendation payload)
        ↓
CanonicalDecisionFlowGate (intergrax/runtime/decision_flow.py)
        ↓
DispositionGovernanceEvaluator → DecisionGovernanceDisposition.ALLOW
        ↓
mint_validated_execution_authorization (intergrax/runtime/decision_authorization.py)
        ↓
DecisionExecutionAuthorization
        ↓
validate_execution_authorization_bundle
        ↓
single_model_inference_execution_request → ExecutionRequest
        ↓
Execution / ExecutionRuntime (StrategyExecutionRouter → InferenceExecutor)
        ↓
ExecutionResult (completed)
        ↓
DecisionExecutionCorrelationRecord + InMemoryDecisionAuditSink integration
```

---

## Governance matrix

| Scenario | Authorization | Canonical Execution Engine |
| --- | --- | --- |
| ALLOW (`canonical-execution-success`) | yes | yes |
| DENY (`canonical-governance-deny`) | no | no |
| REQUIRE_HUMAN (`canonical-governance-approval`) | no | no |

---

## Proof tests

| Layer | Path |
| --- | --- |
| Local (5 scenarios) | `tests/unit/testing_support/decision_e2e/test_canonical_docker_execution.py` |
| Docker in-container | `tests/integration/decision_system/test_docker_e2e_system_qualification.py` (`test_docker_canonical_*`) |

Worker scenarios: `canonical-execution-success`, `canonical-governance-deny`, `canonical-governance-approval`, `canonical-evidence-chain`, `canonical-execution-failure`.

---

## Docker gate (canonical success)

- **Command:** `uv run pytest tests/integration/decision_system/test_docker_e2e_system_qualification.py -k canonical`
- **Image:** `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
- **Requirement:** Docker CLI + running daemon (`pytest.fail` when blocked for canonical gate tests)
