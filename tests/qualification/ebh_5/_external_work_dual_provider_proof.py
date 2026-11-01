# © Artur Czarnecki. All rights reserved.

"""Replaceability proof: two distinct ExternalWorkIntegration implementations via profile binding."""

from __future__ import annotations

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.integrations.registry.profile import IntegrationProfile
from tests.qualification.ebh_5._qualification_external_work_providers import (
    QualExternalWorkAlphaIntegration,
    QualExternalWorkBetaIntegration,
    qual_external_work_create_request,
)


def prove_external_work_dual_profile_binding() -> None:
    alpha = QualExternalWorkAlphaIntegration()
    beta = QualExternalWorkBetaIntegration()
    assert type(alpha) is not type(beta)
    assert isinstance(alpha, ExternalWorkIntegration)
    assert isinstance(beta, ExternalWorkIntegration)

    pa = IntegrationProfile(external_work=alpha)
    pb = IntegrationProfile(external_work=beta)
    ia = pa.instance_for_category(IntegrationCategory.EXTERNAL_WORK)
    ib = pb.instance_for_category(IntegrationCategory.EXTERNAL_WORK)
    assert ia is alpha
    assert ib is beta

    req_a = qual_external_work_create_request(provider_id="ebh7_qual_external_work_alpha")
    req_b = qual_external_work_create_request(
        provider_id="ebh7_qual_external_work_beta",
        idempotency_key="idem-beta-1",
        task_id="task-beta",
    )
    snap_a = ia.create_work(req_a)
    snap_b = ib.create_work(req_b)
    assert snap_a.provider_state_label == "alpha-impl"
    assert snap_b.provider_state_label == "beta-impl"
    assert alpha.discover().identity.provider_id != beta.discover().identity.provider_id

    alpha.unsupported.add("create_work")
    with pytest.raises(Exception):
        alpha.create_work(req_a)
    assert beta.create_work(req_b).provider_state_label == "beta-impl"


def prove_external_work_profile_binding_fail_closed() -> None:
    profile = IntegrationProfile(external_work=object())
    with pytest.raises(Exception):
        profile.instance_for_category(IntegrationCategory.EXTERNAL_WORK)
