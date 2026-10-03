# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R2 security suite A/B/C baseline matrix (R2-D)."""

from __future__ import annotations

from typing import Final

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

CTRL_X_SECURITY_BASELINE_A: Final[str] = "5e9878d71d363542d25ada19cff2590aa37448d3"
CTRL_X_SECURITY_BASELINE_B: Final[str] = "1178d6c7b2f5b027bf30e0bddb37d85ef936f15a"

# Recorded via: uv run pytest -p no:xdist tests/unit/runtime/security -q --tb=no
# at A (worktree), B (START_HEAD), and C (post-R2) — identical 15 failures, same node ids.
CTRL_X_SECURITY_FAILURE_MATRIX: Final[tuple[dict[str, str], ...]] = (
    {
        "node_id": "tests/unit/runtime/security/test_p0_safety_7_sandbox_isolation_fail_closed.py::test_valid_sandbox_reaches_provider",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py::test_default_side_effect_tool_executes_once_despite_retry_policy",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py::test_explicit_retry_safe_side_effect_retries_when_authorized",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py::test_fresh_sandbox_authorization_blocks_retry_when_unavailable",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py::test_side_effect_timeout_does_not_blind_retry",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py::test_plugin_default_side_effect_retry_safety_is_single_attempt",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py::test_idempotency_key_preserved_across_retry_attempts",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_sec_ent.py::test_resolve_encryptor_uses_valid_secrets_store",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_sec_ent.py::test_resolve_encryptor_fails_closed_on_resolution_error",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_sec_ent.py::test_resolve_encryptor_fails_closed_on_conformance_error",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_sec_ent.py::test_harness_envelope_encryptor_not_selected_by_resolver",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_sec_ent.py::test_defense_middleware_blocks_cross_tenant_scope",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_sec_ent.py::test_security_spine_counters_increment",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_sec_planes_evol.py::test_defense_blocked_emits_platform_signal",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
    {
        "node_id": "tests/unit/runtime/security/test_sec_planes_evol.py::test_encryption_denied_emits_platform_signal",
        "a": "FAIL",
        "b": "FAIL",
        "c": "FAIL",
        "classification": "PRE-EXISTING TEST DEBT",
        "r1_r2": "no",
    },
)


def test_r2_sec_baseline_matrix_covers_fifteen_failures() -> None:
    assert len(CTRL_X_SECURITY_FAILURE_MATRIX) == 15
    assert all(row["a"] == row["b"] == row["c"] == "FAIL" for row in CTRL_X_SECURITY_FAILURE_MATRIX)
    assert all(row["classification"] == "PRE-EXISTING TEST DEBT" for row in CTRL_X_SECURITY_FAILURE_MATRIX)
