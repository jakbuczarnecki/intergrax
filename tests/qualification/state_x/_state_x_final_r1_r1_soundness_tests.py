# © Artur Czarnecki. All rights reserved.

"""STATE-X-FINAL-R1-R1 closed-world soundness gates and behavioral evidence manifests."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tests.qualification.state_x._state_x_explicit_mechanism_classifications import (
    EXPLICIT_MECHANISM_CLASSIFICATIONS,
)
from tests.qualification.state_x._state_x_closed_world_durable_state_support import (
    STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY,
    DiscoveredMechanism,
    assert_all_discovered_candidates_explicitly_classified,
    assert_durable_state_discovery_fully_classified,
    assert_no_anonymous_components,
    assert_no_anonymous_providers,
    assert_no_anonymous_reference_implementations,
    assert_no_placeholder_classification_records,
    assert_no_review_queue_records,
    assert_unknown_candidate_is_rejected,
    classify_candidate,
    discover_durable_state_candidates,
)
from tests.qualification.state_x._state_x_final_support import (
    ATOMICITY_MATRIX,
    assert_state_x_final_mechanical_gate,
)
from tests.qualification.state_x._r5_backup_restore_support import (
    assert_state_x_current_backup_restore_completeness,
    assert_frz_rec_08_behavioral_evidence_complete,
)
from tests.qualification.state_x.inventory import (
    CURRENT_STATE_X_FAMILY_IDS,
    STATE_X_FINAL_R1_R1_START_HEAD,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_FOUR_FAILURE_NODEIDS: Final[tuple[str, ...]] = (
    "tests/unit/runtime/execution/deadline_authority/test_harness_02_r1a_qualification.py::test_q19_from_registry_requires_deadline_resolver_with_durable_budget",
    "tests/unit/runtime/execution/deadline_authority/test_harness_02_r1a_qualification.py::test_q11_llm_execute_passes_bounded_timeout_to_resilience",
    "tests/unit/runtime/long_running/test_pba_fix_a_checkpoint_port_consumption.py::test_a5_worker_runtime_accepts_fake_port",
    "tests/unit/runtime/long_running/test_pba_fix_a_checkpoint_port_consumption.py::test_r1_2_bridge_restore_with_fake_reader",
)

F16_BEHAVIORAL_PYTEST_ARGS: Final[tuple[str, ...]] = (
    "tests/unit/runtime/background_execution/test_background_execution_bootstrap.py",
    "tests/unit/runtime/background_execution/test_ue_9a_background_identity_redelivery.py",
    "tests/unit/runtime/background_execution/test_p0c7a_background_terminal_durability.py",
    "tests/unit/runtime/background_execution/test_background_execution_identity.py",
)

F17_BEHAVIORAL_PYTEST_ARGS: Final[tuple[str, ...]] = (
    "tests/unit/runtime/execution/continuation",
)

F18_BEHAVIORAL_PYTEST_ARGS: Final[tuple[str, ...]] = (
    "tests/unit/runtime/execution/deadline_authority/test_harness_02_r1_qualification.py",
    "tests/unit/runtime/execution/deadline_authority/test_harness_02_r1a_qualification.py",
)

F19_BEHAVIORAL_PYTEST_ARGS: Final[tuple[str, ...]] = (
    "tests/unit/runtime/execution/test_delegated_invocation_correlation_durability.py",
    "tests/unit/runtime/execution/test_delegated_execution_status.py",
    "tests/unit/runtime/execution/test_delegated_execution_query.py",
    "tests/unit/runtime/execution/test_delegated_execution_continuation.py",
    "tests/unit/runtime/execution/test_delegated_execution_control.py",
)

F20_BEHAVIORAL_PYTEST_ARGS: Final[tuple[str, ...]] = (
    "tests/unit/runtime/execution/suspended_operation/test_durable_suspended_operation_store.py",
    "tests/unit/runtime/execution/suspended_operation/test_suspended_operation_store.py",
    "tests/unit/runtime/execution/suspended_operation/test_uca6c_r6_r5_9_r2_r1_canonical_reentry_fencing.py",
    "tests/unit/runtime/execution/suspended_operation/test_uca6c_r6_r5_9_r2_r1_r1_claim_authority_propagation.py",
)


@dataclass(frozen=True, slots=True)
class PytestManifestResult:
    exit_code: int
    command: str


def _run_pytest(paths: tuple[str, ...]) -> PytestManifestResult:
    cmd = [
        "uv",
        "run",
        "--with",
        "cryptography",
        "pytest",
        *paths,
        "-p",
        "no:xdist",
        "-q",
    ]
    result = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    command = " ".join(cmd)
    return PytestManifestResult(exit_code=result.returncode, command=command)


def assert_pytest_manifest_green(paths: tuple[str, ...], *, label: str) -> None:
    outcome = _run_pytest(paths)
    assert outcome.exit_code == 0, (
        f"{label} behavioral evidence failed (exit={outcome.exit_code}): {outcome.command}"
    )


def assert_f16_behavioral_evidence_green() -> None:
    assert_pytest_manifest_green(F16_BEHAVIORAL_PYTEST_ARGS, label="SX-F16")


def assert_f17_behavioral_evidence_green() -> None:
    assert_pytest_manifest_green(F17_BEHAVIORAL_PYTEST_ARGS, label="SX-F17")


def assert_f18_behavioral_evidence_green() -> None:
    assert_pytest_manifest_green(F18_BEHAVIORAL_PYTEST_ARGS, label="SX-F18")


def assert_f19_behavioral_evidence_green() -> None:
    assert_pytest_manifest_green(F19_BEHAVIORAL_PYTEST_ARGS, label="SX-F19")


def assert_f20_behavioral_evidence_green() -> None:
    assert_pytest_manifest_green(F20_BEHAVIORAL_PYTEST_ARGS, label="SX-F20")


def assert_original_four_failures_green() -> None:
    outcome = _run_pytest(ORIGINAL_FOUR_FAILURE_NODEIDS)
    assert outcome.exit_code == 0, outcome.command


def assert_behavioral_evidence_manifest_paths_exist(
    paths: tuple[str, ...],
    *,
    label: str,
) -> None:
    for rel in paths:
        target = _REPO_ROOT / rel
        assert target.exists(), f"{label} behavioral manifest missing: {rel}"


def assert_f16_behavioral_manifest_registered() -> None:
    assert_behavioral_evidence_manifest_paths_exist(
        F16_BEHAVIORAL_PYTEST_ARGS,
        label="SX-F16",
    )


def assert_f17_behavioral_manifest_registered() -> None:
    assert_behavioral_evidence_manifest_paths_exist(
        F17_BEHAVIORAL_PYTEST_ARGS,
        label="SX-F17",
    )


def assert_f18_behavioral_manifest_registered() -> None:
    assert_behavioral_evidence_manifest_paths_exist(
        F18_BEHAVIORAL_PYTEST_ARGS,
        label="SX-F18",
    )


def assert_f19_behavioral_manifest_registered() -> None:
    assert_behavioral_evidence_manifest_paths_exist(
        F19_BEHAVIORAL_PYTEST_ARGS,
        label="SX-F19",
    )


def assert_f20_behavioral_manifest_registered() -> None:
    assert_behavioral_evidence_manifest_paths_exist(
        F20_BEHAVIORAL_PYTEST_ARGS,
        label="SX-F20",
    )


def f16_through_f20_behavioral_pytest_aggregate_args() -> tuple[str, ...]:
    return (
        *F16_BEHAVIORAL_PYTEST_ARGS,
        *F17_BEHAVIORAL_PYTEST_ARGS,
        *F18_BEHAVIORAL_PYTEST_ARGS,
        *F19_BEHAVIORAL_PYTEST_ARGS,
        *F20_BEHAVIORAL_PYTEST_ARGS,
    )


def assert_r1r1_final_closure_readiness() -> None:
    assert_durable_state_discovery_fully_classified()
    assert_state_x_final_mechanical_gate()
    assert_state_x_current_backup_restore_completeness()
    assert_frz_rec_08_behavioral_evidence_complete()
    matrix_ids = {row.family_id for row in ATOMICITY_MATRIX}
    for fid in ("SX-F16", "SX-F17", "SX-F18", "SX-F19", "SX-F20"):
        assert fid in matrix_ids


def family_classification_complete(family_id: str, owner_symbol: str) -> None:
    hits = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.family_id == family_id or r.symbol == owner_symbol
    ]
    assert hits, f"missing closed-world records for {family_id}"


def assert_known_provider_controls() -> None:
    kv = next(
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.symbol == "KvBackgroundExecutionIdentityPersistence"
    )
    assert kv.family_id == "SX-F16"
    assert kv.classification.value == "provider_implementation"
    cont = next(
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.symbol == "InMemoryExecutionContinuationStateStore"
    )
    assert cont.family_id == "SX-F17"
    backing = next(
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.symbol == "ExecutionContinuationDurableBacking"
    )
    assert backing.family_id == "SX-F17"
    assert backing.classification.value == "component_of_family"


def assert_unknown_file_candidate_unclassified() -> None:
    candidate = DiscoveredMechanism(
        mechanism_id="path:intergrax/runtime/synthetic/some_new_persistence.py",
        symbol="some_new_persistence",
        paths=("intergrax/runtime/synthetic/some_new_persistence.py",),
        discovery_kind="file",
    )
    assert classify_candidate(candidate) is None


def assert_unknown_inmemory_candidate_unclassified() -> None:
    candidate = DiscoveredMechanism(
        mechanism_id="class:intergrax/runtime/synthetic.py::InMemoryBrandNewExecutionFooStore",
        symbol="InMemoryBrandNewExecutionFooStore",
        paths=("intergrax/runtime/synthetic.py",),
        discovery_kind="class",
    )
    assert classify_candidate(candidate) is None


def assert_outside_requires_explicit_registry_entry() -> None:
    outside_mid = (
        "class:intergrax/runtime/events/persistence_contract.py::NullRuntimeEventPersistence"
    )
    assert outside_mid in EXPLICIT_MECHANISM_CLASSIFICATIONS
    trimmed = {
        k: v for k, v in EXPLICIT_MECHANISM_CLASSIFICATIONS.items() if k != outside_mid
    }
    candidate = DiscoveredMechanism(
        mechanism_id=outside_mid,
        symbol="NullRuntimeEventPersistence",
        paths=("intergrax/runtime/events/persistence_contract.py",),
        discovery_kind="class",
    )
    assert classify_candidate(candidate) is not None
    original = EXPLICIT_MECHANISM_CLASSIFICATIONS
    try:
        EXPLICIT_MECHANISM_CLASSIFICATIONS.clear()
        EXPLICIT_MECHANISM_CLASSIFICATIONS.update(trimmed)
        assert classify_candidate(candidate) is None
    finally:
        EXPLICIT_MECHANISM_CLASSIFICATIONS.clear()
        EXPLICIT_MECHANISM_CLASSIFICATIONS.update(original)


def classification_quality_counts() -> dict[str, int]:
    return {
        "candidates": len(discover_durable_state_candidates()),
        "classified": len(STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY),
        "review_queue": sum(
            1 for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY if r.owner_stage == "REVIEW-QUEUE"
        ),
        "anonymous_components": sum(
            1
            for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
            if r.classification.value == "component_of_family" and r.family_id is None
        ),
        "anonymous_providers": sum(
            1
            for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
            if r.classification.value == "provider_implementation" and r.family_id is None
        ),
        "anonymous_references": sum(
            1
            for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
            if r.classification.value == "non_durable_reference_only" and r.family_id is None
        ),
    }
