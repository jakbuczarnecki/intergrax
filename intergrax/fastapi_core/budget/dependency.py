from fastapi import Depends

from intergrax.fastapi_core.budget.policy import BudgetPolicy
from intergrax.fastapi_core.context import RequestContext, get_request_context
from intergrax.fastapi_core.errors.budget import BudgetExceededError


def require_budget() -> None:
    """
    FastAPI dependency enforcing budget/quota policy.
    """

    def _dependency(
        policy: BudgetPolicy = Depends(BudgetPolicy),
        context: RequestContext = Depends(get_request_context),
    ) -> None:
        if not policy.check_create_run(context):
            raise BudgetExceededError()

    return _dependency
