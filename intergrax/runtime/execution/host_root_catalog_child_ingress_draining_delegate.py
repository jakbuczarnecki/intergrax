# © Artur Czarnecki. All rights reserved.

"""Host root delegate wrapper — cooperative catalog child ingress drain."""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.root_execution_child_work_ingress import (
    RootExecutionChildWorkIngressPort,
)
from intergrax.runtime.execution.root_execution_child_work_ingress import (
    execute_delegate_with_root_child_work_ingress,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class HostRootCatalogChildIngressDrainingDelegate:
    """Runs root business work while draining cross-loop catalog child ingress."""

    __slots__ = ("_inner", "_ingress")

    def __init__(
        self,
        inner: object,
        *,
        ingress: RootExecutionChildWorkIngressPort | None,
    ) -> None:
        self._inner = inner
        self._ingress = ingress

    async def execute(self, request: RequestT) -> ResultT:
        result = await execute_delegate_with_root_child_work_ingress(
            self._inner,
            request,
            self._ingress,
        )
        return result


__all__ = ["HostRootCatalogChildIngressDrainingDelegate"]
