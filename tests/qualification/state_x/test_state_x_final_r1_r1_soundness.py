# © Artur Czarnecki. All rights reserved.

"""STATE-X-FINAL-R1-R1 soundness qualification entrypoint."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.qualification.state_x._state_x_final_r1_r1_soundness_tests import (
    ORIGINAL_FOUR_FAILURE_NODEIDS,
    assert_all_discovered_candidates_explicitly_classified,
    assert_durable_state_discovery_fully_classified,
    assert_f16_behavioral_manifest_registered,
    assert_f17_behavioral_manifest_registered,
    assert_f18_behavioral_manifest_registered,
    assert_f19_behavioral_manifest_registered,
    assert_f20_behavioral_manifest_registered,
    assert_known_provider_controls,
    assert_no_anonymous_components,
    assert_no_anonymous_providers,
    assert_no_anonymous_reference_implementations,
    assert_no_placeholder_classification_records,
    assert_no_review_queue_records,
    assert_outside_requires_explicit_registry_entry,
    assert_r1r1_final_closure_readiness,
    assert_unknown_candidate_is_rejected,
    assert_unknown_file_candidate_unclassified,
    assert_unknown_inmemory_candidate_unclassified,
    classification_quality_counts,
    family_classification_complete,
)
from tests.qualification.state_x._state_x_final_support import assert_state_x_final_mechanical_gate
from tests.qualification.state_x._state_x_final_r1_closed_world_tests import (
    test_sxf_r1_q35_backup_restore_current_complete,
    test_sxf_r1_q36_atomicity_matrix_current,
)
from tests.qualification.state_x.inventory import (
    CURRENT_STATE_X_FAMILY_IDS,
    STATE_X_FINAL_R1_R1_START_HEAD,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_r1r1_q01_start_head_exact() -> None:
    assert STATE_X_FINAL_R1_R1_START_HEAD == "d2b08198c674ba2f40e6a30afadb198fb9a3d5f9"


def test_r1r1_q02_no_review_queue_records() -> None:
    assert_no_review_queue_records()


def test_r1r1_q03_unknown_class_candidate_unclassified() -> None:
    assert_unknown_candidate_is_rejected()


def test_r1r1_q04_unknown_file_candidate_unclassified() -> None:
    assert_unknown_file_candidate_unclassified()


def test_r1r1_q05_unknown_inmemory_candidate_unclassified() -> None:
    assert_unknown_inmemory_candidate_unclassified()


def test_r1r1_q06_no_catch_all_outside() -> None:
    assert_no_review_queue_records()
    assert_no_placeholder_classification_records()


def test_r1r1_q07_component_family_id_mandatory() -> None:
    assert_no_anonymous_components()


def test_r1r1_q08_provider_family_id_mandatory() -> None:
    assert_no_anonymous_providers()


def test_r1r1_q09_reference_family_mapping_mandatory() -> None:
    assert_no_anonymous_reference_implementations()


def test_r1r1_q10_canonical_families_complete() -> None:
    assert_durable_state_discovery_fully_classified()


def test_r1r1_q11_projections_non_authoritative() -> None:
    assert_durable_state_discovery_fully_classified()


def test_r1r1_q12_outside_has_owner_stage() -> None:
    assert_durable_state_discovery_fully_classified()


def test_r1r1_q13_discovered_equals_classified() -> None:
    assert_all_discovered_candidates_explicitly_classified()


def test_r1r1_q14_unclassified_zero() -> None:
    counts = classification_quality_counts()
    assert counts["candidates"] == counts["classified"]


def test_r1r1_q15_future_unknown_store_sentinel() -> None:
    assert_unknown_candidate_is_rejected()


def test_r1r1_q16_outside_requires_explicit_entry() -> None:
    assert_outside_requires_explicit_registry_entry()


def test_r1r1_q17_no_placeholder_owners() -> None:
    assert_no_placeholder_classification_records()


@pytest.mark.parametrize(
    ("family_id", "owner_symbol"),
    [
        ("SX-F16", "BackgroundExecutionIdentityPersistence"),
        ("SX-F17", "ExecutionContinuationStateStore"),
        ("SX-F18", "KvExecutionDeadlinePersistence"),
        ("SX-F19", "DelegatedInvocationCorrelationStore"),
        ("SX-F20", "SuspendedExecutionOperationStore"),
    ],
)
def test_r1r1_families_classified(family_id: str, owner_symbol: str) -> None:
    family_classification_complete(family_id, owner_symbol)


def test_r1r1_known_provider_controls() -> None:
    assert_known_provider_controls()


def test_r1r1_q23_f16_behavioral_manifest() -> None:
    assert_f16_behavioral_manifest_registered()


def test_r1r1_q24_f17_behavioral_manifest() -> None:
    assert_f17_behavioral_manifest_registered()


def test_r1r1_q25_f18_behavioral_manifest() -> None:
    assert_f18_behavioral_manifest_registered()


def test_r1r1_q26_f19_behavioral_manifest() -> None:
    assert_f19_behavioral_manifest_registered()


def test_r1r1_q27_f20_behavioral_manifest() -> None:
    assert_f20_behavioral_manifest_registered()


def test_r1r1_q28_original_four_failures_recorded() -> None:
    assert len(ORIGINAL_FOUR_FAILURE_NODEIDS) == 4


def test_r1r1_q29_baseline_reproduction_performed() -> None:
    baseline = _REPO_ROOT / ".tmp" / "state-x-r1r1-baseline"
    if not baseline.is_dir():
        pytest.skip("baseline worktree not present; operator may add at 4ed4c01d3ce5417fc902ace213d3a4f53c9064bc")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "cryptography",
            "pytest",
            *ORIGINAL_FOUR_FAILURE_NODEIDS,
            "-p",
            "no:xdist",
            "-q",
        ],
        cwd=baseline,
        check=False,
    )
    assert result.returncode != 0


def test_r1r1_q30_original_four_nodeids_recorded() -> None:
    for nodeid in ORIGINAL_FOUR_FAILURE_NODEIDS:
        path_part, _, name = nodeid.partition("::")
        assert (_REPO_ROOT / path_part).is_file(), nodeid
        assert name.startswith("test_")


def test_r1r1_q31_backup_matrix() -> None:
    test_sxf_r1_q35_backup_restore_current_complete()


def test_r1r1_q32_atomicity_matrix() -> None:
    test_sxf_r1_q36_atomicity_matrix_current()


def test_r1r1_q40_parent_suite_imports() -> None:
    assert_state_x_final_mechanical_gate()


def test_r1r1_q43_current_family_registry() -> None:
    assert len(CURRENT_STATE_X_FAMILY_IDS) == 20


def test_r1r1_q50_final_closure_readiness() -> None:
    assert_r1r1_final_closure_readiness()
