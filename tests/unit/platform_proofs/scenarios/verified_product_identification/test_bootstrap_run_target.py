"""Unit tests for bootstrap run-target semantics."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapCompatibilityError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.run_target import (
    assert_requested_target_not_below_checkpoint,
    checkpoint_meets_target,
    manifest_run_target_value,
    resolve_requested_target_rows,
)

pytestmark = pytest.mark.unit


def test_resolve_requested_target_rows_with_cap() -> None:
    assert resolve_requested_target_rows(max_records=1000, dataset_record_count=5000) == 1000


def test_resolve_requested_target_rows_full_dataset() -> None:
    assert resolve_requested_target_rows(max_records=None, dataset_record_count=5000) == 5000


def test_checkpoint_meets_target() -> None:
    assert checkpoint_meets_target(checkpoint_rows_processed=1000, requested_target_rows=1000) is True
    assert checkpoint_meets_target(checkpoint_rows_processed=999, requested_target_rows=1000) is False


def test_requested_target_below_checkpoint_fails_closed() -> None:
    with pytest.raises(VpiBootstrapCompatibilityError, match="below existing checkpoint"):
        assert_requested_target_not_below_checkpoint(
            requested_target_rows=500,
            checkpoint_rows_processed=1000,
        )


def test_manifest_run_target_value_full_scope_is_none() -> None:
    assert manifest_run_target_value(5000, 5000) is None
