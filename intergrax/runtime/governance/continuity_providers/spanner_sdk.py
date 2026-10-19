# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Cloud Spanner client boundary.

Construction refuses before importing or calling the client. Cloud execution
is not authorized, and this module does not qualify Spanner.
"""

from __future__ import annotations

import importlib

from intergrax.runtime.governance.continuity_providers.credentials import (
    OfflineProviderRefusal,
    evaluate_continuity_credentials,
)
from intergrax.runtime.governance.continuity_providers.targets import SpannerPrimaryTarget

_CLOUD_EXECUTION_AUTHORIZED = False


def cloud_execution_authorized() -> bool:
    return _CLOUD_EXECUTION_AUTHORIZED


def open_spanner_sdk_database(target: SpannerPrimaryTarget, binding: object) -> object:
    """Refuse a live session. The client import below is not reached."""

    decision = evaluate_continuity_credentials(binding)
    if (
        not _CLOUD_EXECUTION_AUTHORIZED
        or decision.token is None
        or type(target) is not SpannerPrimaryTarget
    ):
        raise OfflineProviderRefusal(decision.refused.value)
    spanner = importlib.import_module("google.cloud.spanner")
    client = getattr(spanner, "Client")(project=target.project_id)
    database = client.instance(target.instance_id).database(target.database_id)

    def _run(unit):
        return database.run_in_transaction(unit)

    return _run
