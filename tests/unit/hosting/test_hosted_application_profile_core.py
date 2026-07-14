# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Core behavior tests for hosted application profile contracts (APP-HOST-1A.1)."""

from __future__ import annotations

import math

import pytest
from pydantic import SecretStr, ValidationError

from intergrax.hosting import (
    HOSTED_APPLICATION_PROFILE_SPEC_VERSION,
    HostedApplicationProfile,
)

pytestmark = pytest.mark.unit


def sample_application_factory() -> object:
    return object()


def test_minimal_profile_construction() -> None:
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
    )
    assert profile.application_id == "my_application"
    assert profile.application_factory is sample_application_factory
    assert profile.application_factory_id is not None
    assert profile.metadata == {}


def test_default_spec_version_is_1_0() -> None:
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
    )
    assert profile.spec_version == "1.0"
    assert HOSTED_APPLICATION_PROFILE_SPEC_VERSION == "1.0"


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("My_Application", "my_application"),
        ("  my_application  ", "my_application"),
    ],
)
def test_application_id_normalization(raw_id: str, expected: str) -> None:
    profile = HostedApplicationProfile(
        application_id=raw_id,
        application_factory=sample_application_factory,
    )
    assert profile.application_id == expected
    assert profile.identity.application_id == expected


@pytest.mark.parametrize(
    "invalid_id",
    [
        "",
        "1application",
        "my-application",
        "my application",
        "application/path",
    ],
)
def test_invalid_application_id_rejection(invalid_id: str) -> None:
    with pytest.raises(ValidationError):
        HostedApplicationProfile(
            application_id=invalid_id,
            application_factory=sample_application_factory,
        )


def test_stable_factory_id_derived_for_top_level_function() -> None:
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
    )
    expected_factory_id = (
        f"{sample_application_factory.__module__}.{sample_application_factory.__qualname__}"
    )
    assert profile.application_factory_id == expected_factory_id


def test_explicit_factory_id_accepted() -> None:
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
        application_factory_id="package.module.create_application",
    )
    assert profile.application_factory_id == "package.module.create_application"
    assert profile.identity.application_factory_id == "package.module.create_application"


def test_lambda_without_explicit_factory_id_rejected() -> None:
    with pytest.raises(ValidationError, match="application_factory_id"):
        HostedApplicationProfile(
            application_id="my_application",
            application_factory=lambda: None,
        )


def test_local_callable_without_explicit_factory_id_rejected() -> None:
    def _local_factory() -> object:
        return object()

    with pytest.raises(ValidationError, match="application_factory_id"):
        HostedApplicationProfile(
            application_id="my_application",
            application_factory=_local_factory,
        )


def test_lambda_with_explicit_factory_id_accepted() -> None:
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=lambda: None,
        application_factory_id="custom.lambda.factory",
    )
    assert profile.application_factory_id == "custom.lambda.factory"


def test_profile_identity_contains_no_callable() -> None:
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
    )
    identity = profile.identity
    assert identity.application_id == "my_application"
    assert "application_factory" not in identity.model_dump()
    assert identity.model_fields_set == {"application_id", "application_factory_id"}


def test_metadata_accepts_nested_json_values() -> None:
    metadata = {
        "enabled": True,
        "count": 3,
        "ratio": 0.5,
        "label": "local",
        "tags": ["alpha", "beta"],
        "nested": {"mode": "foreground", "retries": None},
    }
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
        metadata=metadata,
    )
    assert profile.metadata == metadata


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {"value": object()},
        {"value": b"secret"},
        {"value": SecretStr("secret")},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": float("-inf")},
    ],
)
def test_metadata_rejects_unsupported_python_objects(invalid_metadata: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        HostedApplicationProfile(
            application_id="my_application",
            application_factory=sample_application_factory,
            metadata=invalid_metadata,  # type: ignore[arg-type]
        )


def test_metadata_is_copied_from_caller_mapping() -> None:
    nested = {"mode": "foreground"}
    metadata = {"nested": nested}
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
        metadata=metadata,  # type: ignore[arg-type]
    )
    nested["mode"] = "mutated"
    assert profile.metadata == {"nested": {"mode": "foreground"}}


def test_extra_profile_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        HostedApplicationProfile(
            application_id="my_application",
            application_factory=sample_application_factory,
            unexpected_field=True,  # type: ignore[call-arg]
        )


def test_profile_field_reassignment_rejected() -> None:
    profile = HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
    )
    with pytest.raises(ValidationError):
        profile.application_id = "other_application"  # type: ignore[misc]


def test_non_finite_float_rejected_via_math_isfinite_path() -> None:
    assert not math.isfinite(float("nan"))
    with pytest.raises(ValidationError):
        HostedApplicationProfile(
            application_id="my_application",
            application_factory=sample_application_factory,
            metadata={"score": float("nan")},
        )
