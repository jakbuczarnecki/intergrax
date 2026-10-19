# © Artur Czarnecki. All rights reserved.

"""Adversarial gates — PROD-Q-R1 strict external-work provider qualification pins."""

from __future__ import annotations

from dataclasses import replace

import pytest

from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.resolve_strict_external_work_provider_capabilities import (
    StrictExternalWorkProviderResolutionError,
    resolve_strict_external_work_provider_capabilities,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from tests.qualification.prod_q._prod_q_strict_external_work_provider_qualification_config import (
    QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST,
    QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_ID,
    assert_qualified_pin_resolves_against_integration,
    assert_unpinned_strict_external_work_settings_fail_closed,
    with_qualified_strict_external_work_provider_pin,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_qualified_pin_matches_deterministic_fake_discover() -> None:
    fake = DeterministicExternalWorkFake()
    assert_qualified_pin_resolves_against_integration(fake)


def test_missing_provider_id_fail_closed() -> None:
    fake = DeterministicExternalWorkFake()
    assert_unpinned_strict_external_work_settings_fail_closed(fake)


def test_unknown_provider_id_pin_mismatch() -> None:
    fake = DeterministicExternalWorkFake()
    settings = with_qualified_strict_external_work_provider_pin(
        GovernedContractorBackendSettings(),
    )
    settings = replace(settings, strict_external_work_provider_id="unknown-provider")
    with pytest.raises(
        StrictExternalWorkProviderResolutionError,
        match="provider_id pin mismatch",
    ):
        resolve_strict_external_work_provider_capabilities(fake, settings=settings)


def test_wrong_descriptor_digest_pin_mismatch() -> None:
    fake = DeterministicExternalWorkFake()
    settings = GovernedContractorBackendSettings(
        strict_external_work_provider_id=QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_ID,
        strict_external_work_provider_descriptor_digest="sha256:" + ("ff" * 32),
    )
    with pytest.raises(
        StrictExternalWorkProviderResolutionError,
        match="descriptor digest pin mismatch",
    ):
        resolve_strict_external_work_provider_capabilities(fake, settings=settings)


def test_qualified_constants_track_deterministic_fake_digest() -> None:
    descriptor = DeterministicExternalWorkFake().discover()
    assert descriptor.identity.provider_id == QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_ID
    assert descriptor.descriptor_digest == (
        QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST
    )
