# © Artur Czarnecki. All rights reserved.

"""Unit tests for proof-local dataset identity and fingerprint."""

from __future__ import annotations

import pytest

from platform_proofs.tools.iterative_sql_investigation.dataset import PROOF_ROW_COUNT
from platform_proofs.tools.iterative_sql_investigation.dataset_identity import (
    DATASET_ID,
    DATASET_VERSION,
    GROUND_TRUTH_VERSION,
    DatasetIdentity,
    compute_dataset_fingerprint,
)

pytestmark = pytest.mark.unit


def test_canonical_identity_values() -> None:
    identity = DatasetIdentity.canonical()
    assert identity.dataset_id == DATASET_ID
    assert identity.dataset_version == DATASET_VERSION
    assert identity.seed == 42
    assert identity.row_count == PROOF_ROW_COUNT
    assert identity.ground_truth_version == GROUND_TRUTH_VERSION


def test_fingerprint_is_deterministic() -> None:
    first = compute_dataset_fingerprint()
    second = compute_dataset_fingerprint()
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
