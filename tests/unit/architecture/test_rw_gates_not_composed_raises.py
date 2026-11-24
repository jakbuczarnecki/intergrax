"""NT-RW-GATES-NC-RAISE — NOT_COMPOSED trust admission must not pass through when enforced."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from intergrax.applications._shared.deploy_activate_pipeline import (
    DeployActivatePipelineContext,
    RevisionTrustAdmissionNotComposedError,
    RevisionTrustAdmissionState,
    _resolve_revision_trust_admission,
    assert_composed_revision_trust_admission,
)

pytestmark = pytest.mark.unit


def test_not_composed_trust_admission_raises_when_enforced() -> None:
    composition = MagicMock()
    composition.agent_capability_runtime = None
    ctx = DeployActivatePipelineContext(
        composition=composition,
        services=MagicMock(),
        mutation_authorization_boundary=MagicMock(),
        environment_tenant_resolver=MagicMock(),
        require_composed_trust_admission=True,
    )
    _hook, state = _resolve_revision_trust_admission(ctx)
    assert state is RevisionTrustAdmissionState.NOT_COMPOSED
    with pytest.raises(RevisionTrustAdmissionNotComposedError):
        assert_composed_revision_trust_admission(ctx, state)


def test_not_composed_pass_through_when_not_enforced() -> None:
    composition = MagicMock()
    composition.agent_capability_runtime = None
    ctx = DeployActivatePipelineContext(
        composition=composition,
        services=MagicMock(),
        mutation_authorization_boundary=MagicMock(),
        environment_tenant_resolver=MagicMock(),
        require_composed_trust_admission=False,
    )
    _hook, state = _resolve_revision_trust_admission(ctx)
    assert_composed_revision_trust_admission(ctx, state)
