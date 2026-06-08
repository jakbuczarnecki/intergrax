# © Artur Czarnecki. All rights reserved.

"""Critic orchestrator — L0→L1→L2 pipeline with short-circuit (Phase CRIT-V-3.1)."""

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticRequest,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
)
from intergrax.runtime.critic.l0_gateway import L0Gateway
from intergrax.runtime.critic.l1_gateway import L1Gateway

_LAYER_PIPELINE: tuple[CriticLayer, ...] = (
    CriticLayer.L0_DETERMINISTIC,
    CriticLayer.L1_SEMANTIC,
    CriticLayer.L1_TRAJECTORY,
    CriticLayer.L2_HUMAN,
)


class CriticOrchestrator:
    """
    Single CVL entry point for partial and final verification.

    Runs enabled layers in order and short-circuits on hard failure.
    Never calls LLM directly — L1 delegates to Tier-0 tools via ``L1Gateway``.
    """

    def __init__(
        self,
        *,
        l0_gateway: L0Gateway | None = None,
        l1_gateway: L1Gateway | None = None,
    ) -> None:
        self._l0 = l0_gateway or L0Gateway()
        self._l1 = l1_gateway or L1Gateway()

    @property
    def l1_client_configured(self) -> bool:
        return self._l1.client_configured

    def verify(
        self,
        request: CriticRequest,
        *,
        contract: AgentContract | None = None,
    ) -> CriticVerdict:
        resolved_contract = _resolve_contract(contract, request)
        capability = _optional_str(request.context.get("capability"))
        plan_criteria = _optional_str_list(request.context.get("plan_criteria"))
        enabled = frozenset(request.enabled_layers)

        layer_results: list[LayerVerdict] = []
        for layer in _LAYER_PIPELINE:
            if layer not in enabled:
                continue
            verdict = self._run_layer(
                layer,
                request,
                contract=resolved_contract,
                capability=capability,
                plan_criteria=plan_criteria,
            )
            layer_results.append(verdict)
            if not verdict.passed:
                return _build_verdict(
                    request.scope,
                    layer_results,
                    passed=False,
                    failed_layer=layer,
                )

        return _build_verdict(request.scope, layer_results, passed=True, failed_layer=None)

    def verify_partial(
        self,
        request: CriticRequest,
        *,
        contract: AgentContract | None = None,
    ) -> CriticVerdict:
        scoped = _with_scope(request, CriticScope.NODE_PARTIAL)
        return self.verify(scoped, contract=contract)

    def verify_final(
        self,
        request: CriticRequest,
        *,
        contract: AgentContract | None = None,
    ) -> CriticVerdict:
        scoped = _with_scope(request, CriticScope.GRAPH_FINAL)
        return self.verify(scoped, contract=contract)

    def _run_layer(
        self,
        layer: CriticLayer,
        request: CriticRequest,
        *,
        contract: AgentContract,
        capability: str | None,
        plan_criteria: list[str] | None,
    ) -> LayerVerdict:
        if layer is CriticLayer.L0_DETERMINISTIC:
            return self._l0.verify(
                request,
                contract=contract,
                capability=capability,
                plan_criteria=plan_criteria,
            )
        if layer is CriticLayer.L1_SEMANTIC:
            return self._l1.verify_semantic(request)
        if layer is CriticLayer.L1_TRAJECTORY:
            return self._l1.verify_trajectory(request)
        return LayerVerdict(
            layer=CriticLayer.L2_HUMAN,
            passed=False,
            errors=["L2 human verification is not wired; escalate via HITL policy"],
        )


def _resolve_contract(contract: AgentContract | None, request: CriticRequest) -> AgentContract:
    if contract is not None:
        return contract
    raw = request.context.get("contract")
    if isinstance(raw, AgentContract):
        return raw
    return AgentContract(
        id=request.agent_id,
        name=request.agent_id,
        description="critic default contract",
    )


def _with_scope(request: CriticRequest, scope: CriticScope) -> CriticRequest:
    from dataclasses import replace

    return replace(request, scope=scope)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _optional_str_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [item for item in value if isinstance(item, str)]
    return items or None


def _recommended_action(scope: CriticScope, failed_layer: CriticLayer | None) -> CriticAction:
    if failed_layer is None:
        return CriticAction.CONTINUE
    if failed_layer is CriticLayer.L0_DETERMINISTIC:
        return CriticAction.RETRY if scope is CriticScope.NODE_PARTIAL else CriticAction.FAIL
    if failed_layer in (CriticLayer.L1_SEMANTIC, CriticLayer.L1_TRAJECTORY):
        return CriticAction.REVISE
    if failed_layer is CriticLayer.L2_HUMAN:
        return CriticAction.ESCALATE_HITL
    return CriticAction.FAIL


def _build_verdict(
    scope: CriticScope,
    layers: list[LayerVerdict],
    *,
    passed: bool,
    failed_layer: CriticLayer | None,
) -> CriticVerdict:
    failure_reasons: list[str] = []
    for layer in layers:
        if not layer.passed:
            failure_reasons.extend(layer.errors)
    return CriticVerdict(
        scope=scope,
        passed=passed,
        layers=layers,
        recommended_action=_recommended_action(scope, failed_layer),
        failure_reasons=list(dict.fromkeys(failure_reasons)),
    )
