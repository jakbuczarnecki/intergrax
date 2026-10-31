# © Artur Czarnecki. All rights reserved.

"""Dual in-process ExternalWorkIntegration providers for EBH-7 RPL (profile binding, not catalog slug)."""

from __future__ import annotations

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.integrations.registry.profile import IntegrationProfile
from tests.unit.integrations.test_external_work_integration import _InMemoryExternalWorkIntegration


def prove_external_work_dual_profile_binding() -> None:
    alpha = _InMemoryExternalWorkIntegration()
    beta = _InMemoryExternalWorkIntegration()
    assert isinstance(alpha, ExternalWorkIntegration)
    assert isinstance(beta, ExternalWorkIntegration)
    pa = IntegrationProfile(external_work=alpha)
    pb = IntegrationProfile(external_work=beta)
    ia = pa.instance_for_category(IntegrationCategory.EXTERNAL_WORK)
    ib = pb.instance_for_category(IntegrationCategory.EXTERNAL_WORK)
    assert ia is alpha
    assert ib is beta
    assert ia is not ib


def prove_external_work_profile_binding_fail_closed() -> None:
    profile = IntegrationProfile(external_work=object())
    with pytest.raises(Exception):
        profile.instance_for_category(IntegrationCategory.EXTERNAL_WORK)
