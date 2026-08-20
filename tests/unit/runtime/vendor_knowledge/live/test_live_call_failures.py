# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.vendor_knowledge.live.errors import LiveErrorCodeV1
from intergrax.runtime.vendor_knowledge.live.failures import (
    LiveCallFailureReasonV1,
    live_call_failure_reason_for_error_code,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        (LiveErrorCodeV1.BINDING_UNAVAILABLE.value, LiveCallFailureReasonV1.AUTHORITY_UNAVAILABLE),
        (
            LiveErrorCodeV1.PROVIDER_TEMPORARILY_UNAVAILABLE.value,
            LiveCallFailureReasonV1.PROVIDER_FAILED,
        ),
        (
            LiveErrorCodeV1.PROVIDER_CONTRACT_VIOLATION.value,
            LiveCallFailureReasonV1.PROVIDER_RESPONSE_INVALID,
        ),
        (LiveErrorCodeV1.RESULT_INVALID.value, LiveCallFailureReasonV1.PROVIDER_RESPONSE_INVALID),
    ],
)
def test_live_call_failure_reason_mapping(error_code: str, expected: LiveCallFailureReasonV1) -> None:
    assert live_call_failure_reason_for_error_code(error_code) is expected
