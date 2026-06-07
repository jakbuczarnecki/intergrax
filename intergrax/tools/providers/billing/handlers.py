# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.billing.contracts import (
    BillingListUsageInput,
    BillingListUsageOutput,
    BillingRecordUsageInput,
    BillingRecordUsageOutput,
)
from intergrax.tools.providers.billing.service import billing_list_usage, billing_record_usage


class BillingRecordUsageHandler(ServiceToolHandler[BillingRecordUsageInput, BillingRecordUsageOutput]):
    _service = billing_record_usage


class BillingListUsageHandler(ServiceToolHandler[BillingListUsageInput, BillingListUsageOutput]):
    _service = billing_list_usage
