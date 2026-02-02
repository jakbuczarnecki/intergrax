# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.fastapi_core.budget.policy import BudgetPolicy
from intergrax.fastapi_core.context import RequestContext


class NoOpBudgetPolicy(BudgetPolicy):
    def check_create_run(self, context: RequestContext) -> bool:
        return True
