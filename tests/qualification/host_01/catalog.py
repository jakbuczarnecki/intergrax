# © Artur Czarnecki. All rights reserved.

"""HOST-01 Q1..Q12 evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Host01QEvidence:
    q_id: str
    title: str
    pytest_node_ids: tuple[str, ...]


def _nid(test_file: str, test_name: str) -> str:
    return f"tests/qualification/host_01/{test_file}::{test_name}"


HOST_01_Q_CATALOG: tuple[Host01QEvidence, ...] = (
    Host01QEvidence(
        "HOST-Q1",
        "Discovered hosts reach canonical application boundary",
        (
            _nid("test_host_01_gates.py", "test_host_q1_production_surfaces_use_host_task_execution_port"),
            _nid(
                "test_host_01_gates.py",
                "test_host_threaded_execution_adapter_not_wired_in_production_composition",
            ),
        ),
    ),
    Host01QEvidence(
        "HOST-Q2",
        "No host bypasses governance launch path",
        (_nid("test_host_01_gates.py", "test_host_q2_host_task_roots_on_root_execution_launcher"),),
    ),
    Host01QEvidence(
        "HOST-Q3",
        "Identity resolved via platform root execution context",
        (_nid("test_host_01_gates.py", "test_host_q3_host_task_resolves_root_execution_context"),),
    ),
    Host01QEvidence(
        "HOST-Q4",
        "Shared Execution facade/runtime path",
        (_nid("test_host_01_gates.py", "test_host_q4_host_task_uses_execution_runtime"),),
    ),
    Host01QEvidence(
        "HOST-Q5",
        "Canonical error semantics with transport mapping isolated",
        (_nid("test_host_01_gates.py", "test_host_q5_fastapi_maps_internal_errors_without_secret_leak"),),
    ),
    Host01QEvidence(
        "HOST-Q6",
        "Transport types absent from canonical execution contracts",
        (_nid("test_host_01_gates.py", "test_host_q6_canonical_execution_request_has_no_transport_imports"),),
    ),
    Host01QEvidence(
        "HOST-Q7",
        "Custom host adapter via public port",
        (_nid("test_host_01_gates.py", "test_host_q7_custom_adapter_delegates_without_core_change"),),
    ),
    Host01QEvidence(
        "HOST-Q8",
        "No prohibited integration patterns on host adapter surface",
        (_nid("test_host_01_gates.py", "test_host_q8_host_adapter_modules_static_gate"),),
    ),
    Host01QEvidence(
        "HOST-Q9",
        "Single execution side effect per host invocation",
        (_nid("test_host_01_gates.py", "test_host_q9_single_facade_invoke_per_adapter_path"),),
    ),
    Host01QEvidence(
        "HOST-Q10",
        "Semantic equivalence across HTTP harness and MCP intake",
        (_nid("test_host_01_gates.py", "test_host_q10_mcp_and_http_harness_map_equivalent_task_semantics"),),
    ),
    Host01QEvidence(
        "HOST-Q11",
        "Safe error exposure at HTTP host boundary",
        (_nid("test_host_01_gates.py", "test_host_q11_global_handler_hides_stack_and_secrets"),),
    ),
    Host01QEvidence(
        "HOST-Q12",
        "Host adapters do not import vendor/model/persistence implementations",
        (_nid("test_host_01_gates.py", "test_host_q12_host_adapter_import_layer_gate"),),
    ),
)
