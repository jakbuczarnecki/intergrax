# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plugin registry for capability acquisition strategies (UCA-3)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.acquisition_strategy import (
    CapabilityAcquisitionStrategy,
)
from intergrax.contracts.capability_acquisition.errors import (
    CapabilityAcquisitionConfigurationError,
    CapabilityAcquisitionIntegrityError,
)
from intergrax.contracts.capability_acquisition.strategy_descriptor import (
    CapabilityAcquisitionStrategyDescriptor,
)


def descriptor_for_strategy(
    strategy: CapabilityAcquisitionStrategy,
) -> CapabilityAcquisitionStrategyDescriptor:
    return CapabilityAcquisitionStrategyDescriptor(
        strategy_id=strategy.strategy_id,
        supported_kinds=tuple(sorted(strategy.supported_kinds, key=lambda k: k.value)),
    )


class CapabilityAcquisitionStrategyRegistry:
    """Stores strategies and resolves eligibility — not a policy engine."""

    def __init__(
        self,
        strategies: tuple[CapabilityAcquisitionStrategy, ...],
    ) -> None:
        strategy_ids: set[str] = set()
        ordered: list[CapabilityAcquisitionStrategy] = []
        for strategy in strategies:
            strategy_id = strategy.strategy_id
            if strategy_id in strategy_ids:
                raise CapabilityAcquisitionConfigurationError(
                    f"duplicate strategy_id: {strategy_id!r}",
                )
            strategy_ids.add(strategy_id)
            ordered.append(strategy)
        self._strategies = tuple(ordered)
        self._by_id = {strategy.strategy_id: strategy for strategy in self._strategies}

    @property
    def strategies(self) -> tuple[CapabilityAcquisitionStrategy, ...]:
        return self._strategies

    def get(self, strategy_id: str) -> CapabilityAcquisitionStrategy | None:
        return self._by_id.get(strategy_id)

    def eligible_strategies(
        self,
        request: CapabilityAcquisitionRequest,
    ) -> tuple[CapabilityAcquisitionStrategy, ...]:
        """Strategies that support the request with consistent declared kinds."""
        eligible: list[CapabilityAcquisitionStrategy] = []
        for strategy in self._strategies:
            if not strategy.supports(request):
                continue
            _assert_supports_aligned_with_declared_kinds(strategy, request)
            eligible.append(strategy)
        return tuple(eligible)

    def eligible_descriptors(
        self,
        request: CapabilityAcquisitionRequest,
    ) -> tuple[CapabilityAcquisitionStrategyDescriptor, ...]:
        """Descriptors for eligible strategies (invokes ``supports()`` once per strategy)."""
        eligible = self.eligible_strategies(request)
        return tuple(descriptor_for_strategy(strategy) for strategy in eligible)

    def descriptors_for_eligible(
        self,
        eligible: tuple[CapabilityAcquisitionStrategy, ...],
    ) -> tuple[CapabilityAcquisitionStrategyDescriptor, ...]:
        """Build descriptors from a prior eligibility snapshot — no ``supports()`` calls."""
        return tuple(descriptor_for_strategy(strategy) for strategy in eligible)


def _assert_supports_aligned_with_declared_kinds(
    strategy: CapabilityAcquisitionStrategy,
    request: CapabilityAcquisitionRequest,
) -> None:
    declared = strategy.supported_kinds
    if not declared:
        return
    need = request.capability_need
    if need is None or not need.kinds:
        return
    for kind in need.kinds:
        if kind not in declared:
            raise CapabilityAcquisitionIntegrityError(
                f"strategy {strategy.strategy_id!r} supports request but "
                f"declared kinds omit {kind.value!r}",
            )


__all__ = [
    "CapabilityAcquisitionStrategyRegistry",
    "descriptor_for_strategy",
]
