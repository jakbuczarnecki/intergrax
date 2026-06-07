# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.pagerduty.contracts import (
    PagerDutyAcknowledgeIncidentInput,
    PagerDutyAcknowledgeIncidentOutput,
    PagerDutyTriggerIncidentInput,
    PagerDutyTriggerIncidentOutput,
)
from intergrax.tools.providers.pagerduty.service import pagerduty_acknowledge_incident, pagerduty_trigger_incident


class PagerDutyTriggerIncidentHandler(
    ServiceToolHandler[PagerDutyTriggerIncidentInput, PagerDutyTriggerIncidentOutput]
):
    _service = pagerduty_trigger_incident


class PagerDutyAcknowledgeIncidentHandler(
    ServiceToolHandler[PagerDutyAcknowledgeIncidentInput, PagerDutyAcknowledgeIncidentOutput]
):
    _service = pagerduty_acknowledge_incident
