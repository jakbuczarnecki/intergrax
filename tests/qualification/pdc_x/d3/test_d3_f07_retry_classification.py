# © Artur Czarnecki. All rights reserved.

"""PDC-D3 F-D3-07 (defect part) — retry and failover classify by whole words, not by substrings.

``GenerateContentError`` contains ``rate`` ("gene-rate") and ``SeparateTenantError`` contains it too
("sepa-rate"); both used to count as transient, so a non-transient request error was retried and the
same request was re-sent to the secondary provider. The typed provider-error taxonomy itself is an
excluded finding (missing feature); only the substring defect of the existing classifier is repaired.
"""

from __future__ import annotations

import pytest
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.retry import is_retriable_provider_error
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters.registry.failover_policy import default_failover_policy
from tests.qualification.pdc_x.d3.fakes import USER, BufferedOnlyAdapter, FailingAdapter

pytestmark = pytest.mark.unit

_CONFIG = LLMCallConfig(max_retries=2)


def _error(name: str, *, status: int | None = None) -> BaseException:
    attrs: dict[str, object] = {} if status is None else {"status_code": status}
    return type(name, (Exception,), attrs)("boom")


@pytest.mark.parametrize(
    "name",
    [
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "ReadTimeout",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "OverloadedError",
        "ServiceOverloadedException",
        "RateLimitExceeded",
        "rate_limit_error",
    ],
)
def test_transient_class_names_stay_retriable(name: str) -> None:
    assert is_retriable_provider_error(_error(name), _CONFIG) is True


@pytest.mark.parametrize(
    "name",
    ["GenerateContentError", "SeparateTenantError", "MigrateError", "AccelerateError", "ModerateError", "ValueError"],
)
def test_names_that_merely_contain_a_transient_substring_are_not_retriable(name: str) -> None:
    assert is_retriable_provider_error(_error(name), _CONFIG) is False


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_configured_status_codes_stay_retriable_whatever_the_class_name(status: int) -> None:
    assert is_retriable_provider_error(_error("GenerateContentError", status=status), _CONFIG) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_error_statuses_are_not_retriable(status: int) -> None:
    assert is_retriable_provider_error(_error("InvalidRequest", status=status), _CONFIG) is False


@pytest.mark.parametrize("name", ["GenerateContentError", "SeparateTenantError"])
def test_default_failover_does_not_forward_a_non_transient_error_to_the_secondary(name: str) -> None:
    primary = FailingAdapter(_error(name), provider="primary")
    secondary = BufferedOnlyAdapter(provider="secondary")
    chain = FailoverLLMAdapter([primary, secondary], failover_policy=default_failover_policy())

    with pytest.raises(Exception, match="boom"):
        chain.generate_messages(USER)

    assert primary.calls == ["generate_messages"]
    assert secondary.calls == [], "the request content must not reach the secondary provider"


def test_default_failover_still_fails_over_on_a_transient_error() -> None:
    primary = FailingAdapter(_error("APITimeoutError"), provider="primary")
    secondary = BufferedOnlyAdapter(provider="secondary")
    chain = FailoverLLMAdapter([primary, secondary], failover_policy=default_failover_policy())

    response = chain.generate_messages(USER)

    assert response.content == "buffered"
    assert secondary.calls == ["generate_messages"]
