# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.pagerduty.contracts import PagerDutyTriggerIncidentInput, PagerDutyTriggerIncidentOutput
from intergrax.tools.providers.pagerduty.service import pagerduty_trigger_incident


class PagerDutyTriggerIncidentHandler(
    ServiceToolHandler[PagerDutyTriggerIncidentInput, PagerDutyTriggerIncidentOutput]
):
    _service = pagerduty_trigger_incident
