# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition helpers for qualified capability canonical execution (UCA-6C-R2)."""

from __future__ import annotations

from intergrax.contracts.execution.qualified_capability_execution_intake import (
    QualifiedCapabilityExecutionDelegateResult,
    QualifiedCapabilityExecutionIntakePayload,
)
from intergrax.contracts.root_execution_launch import RootExecutionLaunchPort
from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionPort,
)
from intergrax.runtime.execution.canonical_intake_adapter import (
    CanonicalExecutionRuntimeAdapter,
)
from intergrax.runtime.execution.qualified_capability_execution_dispatch_service import (
    QualifiedCapabilityExecutionDispatchService,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.execution.qualified_capability_execution_runtime_delegate import (
    QualifiedCapabilityExecutionRuntimeDelegate,
)
from intergrax.runtime.execution.runtime import ExecutionRuntime
from intergrax.runtime.governance.execution_admission_composition import (
    build_default_root_execution_launcher,
)


def build_qualified_capability_execution_dispatch_service(
    *,
    handler_registry: QualifiedCapabilityExecutionBindingHandlerRegistry,
    runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort,
) -> tuple[
    QualifiedCapabilityExecutionDispatchService,
    QualifiedCapabilityExecutionRuntimeDelegate,
    RootExecutionLaunchPort[
        QualifiedCapabilityExecutionIntakePayload,
        QualifiedCapabilityExecutionDelegateResult,
    ],
]:
    """Wire ingress dedup → root launcher → ExecutionRuntime delegate."""
    delegate = QualifiedCapabilityExecutionRuntimeDelegate(
        handler_registry=handler_registry,
    )
    runtime = ExecutionRuntime(delegate)
    intake = CanonicalExecutionRuntimeAdapter(runtime)
    launcher = build_default_root_execution_launcher(
        runtime_policy_admission=runtime_policy_admission,
        execution_intake=intake,
    )
    dispatch = QualifiedCapabilityExecutionDispatchService(
        root_execution_launcher=launcher,
        runtime_delegate=delegate,
    )
    return dispatch, delegate, launcher


__all__ = ["build_qualified_capability_execution_dispatch_service"]
