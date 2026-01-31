# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.fastapi_core.context import RequestContext


class BudgetPolicy(ABC):
    """
    Control-plane policy limiting resource usage
    (runs, execution capacity, quotas).

    This policy operates BEFORE execution layer (RunStore).
    """

    @abstractmethod
    def check_create_run(self, context: RequestContext) -> bool:
        """
        Validate whether a new run can be created.

        Raises an exception if budget exceeded.
        """
        raise NotImplementedError
