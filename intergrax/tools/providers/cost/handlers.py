# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.cost.contracts import (
    CostCheckQuotaInput,
    CostCheckQuotaOutput,
    CostForecastSpendInput,
    CostForecastSpendOutput,
    CostGetRunBudgetInput,
    CostGetRunBudgetOutput,
)
from intergrax.tools.providers.cost.service import cost_check_quota, cost_forecast_spend, cost_get_run_budget


class CostGetRunBudgetHandler(ServiceToolHandler[CostGetRunBudgetInput, CostGetRunBudgetOutput]):
    _service = cost_get_run_budget


class CostCheckQuotaHandler(ServiceToolHandler[CostCheckQuotaInput, CostCheckQuotaOutput]):
    _service = cost_check_quota


class CostForecastSpendHandler(ServiceToolHandler[CostForecastSpendInput, CostForecastSpendOutput]):
    _service = cost_forecast_spend
