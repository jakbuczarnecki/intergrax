# © Artur Czarnecki. All rights reserved.

"""Qualification-owned contracts for DG-001D R4 supervisor pre-engine failure injection."""

from __future__ import annotations

from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.supervisor.supervisor import HostedApplicationSupervisorLaunchContext

_QUALIFICATION_SECRET_SENTINEL = "DG001D-R4-SECRET-SENTINEL"


def qualification_secret_sentinel() -> str:
    """Return the qualification-only sentinel used in thrown engine-factory exceptions."""
    return _QUALIFICATION_SECRET_SENTINEL


class ControlledFailingHostedApplicationEngineFactory:
    """Inject a deterministic pre-engine failure through the public engine-factory seam."""

    def __call__(
        self,
        launch: HostedApplicationSupervisorLaunchContext,
    ) -> HostedApplicationEngine:
        del launch
        raise RuntimeError(_QUALIFICATION_SECRET_SENTINEL)


def controlled_failing_hosted_application_engine_factory() -> ControlledFailingHostedApplicationEngineFactory:
    return ControlledFailingHostedApplicationEngineFactory()
