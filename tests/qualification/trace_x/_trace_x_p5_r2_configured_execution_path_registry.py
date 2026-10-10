# © Artur Czarnecki. All rights reserved.

"""Explicit TRACE-X-P5-R2 closed-world configured/effective execution path registry."""

from __future__ import annotations

from typing import Final

from tests.qualification.trace_x._trace_x_p5_r2_closed_world_types import (
    ConfiguredExecutionPathClass,
    RegisteredConfiguredExecutionPath,
)

_EVIDENCE = (
    "test_trace_x_p5_r2_closed_world_gates.py::test_txp5cw_q03_configured_execution_paths_closed_world_parity"
)


def _row(
    path: str,
    classification: ConfiguredExecutionPathClass,
    summary: str,
    surface_id: str = "module",
) -> RegisteredConfiguredExecutionPath:
    return RegisteredConfiguredExecutionPath(
        path=path,
        surface_id=surface_id,
        classification=classification,
        summary=summary,
        evidence_nodeid=_EVIDENCE,
    )


CONFIGURED_EXECUTION_PATH_REGISTRY: Final[tuple[RegisteredConfiguredExecutionPath, ...]] = (
    _row(
        "intergrax/applications/_shared/uca6c_marketplace_qualified_execution_composition.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Sanctioned production configured marketplace qualified execution composition root",
    ),
    _row(
        "intergrax/applications/_shared/integrations/persistence.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Durable ExecutionIntegrationConfigurationPinningStore wiring (P2)",
    ),
    _row(
        "intergrax/applications/_shared/integrations/integration_configuration_provenance_reader.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Read-only PinRecord provenance reader for reconstruction (authority=0 for mutation)",
    ),
    _row(
        "intergrax/applications/_shared/integrations/integration_configuration_provenance_requirement_commit.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Production requirement spine commit port adapter",
    ),
    _row(
        "intergrax/autonomous_work/configured_capability_execution_subject_builder.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Typed configured execution subject construction",
    ),
    _row(
        "intergrax/autonomous_work/worker_configured_capability_fulfillment_service.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Explicit opportunity→adoption fulfillment (no global registry)",
    ),
    _row(
        "intergrax/autonomous_work/worker_configured_capability_execution_fulfillment_service.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Configured capability execution fulfillment orchestration",
    ),
    _row(
        "intergrax/contracts/autonomous_work/worker_configured_capability_execution.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Typed configured execution fulfillment contract",
    ),
    _row(
        "intergrax/contracts/autonomous_work/worker_configured_capability_fulfillment.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Typed configured adoption fulfillment contract",
    ),
    _row(
        "intergrax/contracts/autonomous_work/worker_qualified_capability_resume.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Resume contract carrying optional explicit adoption (no ambient reconstruction)",
    ),
    _row(
        "intergrax/contracts/execution/execution_bound_capability_execution_dispatch.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Execution-bound dispatch contract with adoption field",
    ),
    _row(
        "intergrax/contracts/execution/execution_bound_capability_execution_intake.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Execution admission intake with optional adoption",
    ),
    _row(
        "intergrax/contracts/execution/qualified_capability_execution_dispatch.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Qualified capability dispatch contract",
    ),
    _row(
        "intergrax/contracts/execution/qualified_capability_execution_intake.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Qualified capability intake contract",
    ),
    _row(
        "intergrax/contracts/execution_integration_configuration_provenance.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Configured/effective provenance typed contract",
    ),
    _row(
        "intergrax/contracts/execution_integration_configuration_provenance_requirement.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Requirement spine typed contract (INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED)",
    ),
    _row(
        "intergrax/contracts/runtime_event_type.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Runtime event type enum including requirement spine event",
    ),
    _row(
        "intergrax/integrations/configured_relational_store_execution_binding.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Canonical Pattern-A configured relational binding factory",
    ),
    _row(
        "intergrax/integrations/contracts/execution_integration_configuration.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Adoption/opportunity/effective identity contracts",
    ),
    _row(
        "intergrax/integrations/contracts/execution_integration_configuration_pin_record.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "PinRecord codec contract",
    ),
    _row(
        "intergrax/integrations/contracts/execution_integration_configuration_pinning.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "ExecutionIntegrationConfigurationPinningStore contract",
    ),
    _row(
        "intergrax/integrations/execution_bound_configured_relational_store_port.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Execution-bound configured relational I/O port (pin→spine→I/O ordering)",
    ),
    _row(
        "intergrax/integrations/execution_bound_integration_resolution.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Canonical configured provider resolver (ExecutionBoundIntegrationResolution)",
    ),
    _row(
        "intergrax/integrations/execution_integration_configuration_pin_reconciliation.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Pin reconcile semantics (ambiguous outcome / first-writer)",
    ),
    _row(
        "intergrax/integrations/execution_integration_configuration_requirement_fact.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Requirement fact staging validation for CONFIGURED_ADOPTED",
    ),
    _row(
        "intergrax/runtime/execution/integration_configuration_provenance_requirement_recorder.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Single production requirement spine emitter",
    ),
    _row(
        "intergrax/runtime/execution/qualified_capability_execution_handlers.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Handler registry for qualified capability execution",
    ),
    _row(
        "intergrax/runtime/events/payloads/spine_families.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Typed requirement spine payload family",
    ),
    _row(
        "intergrax/tools/configured_integration_tool_invocation_projection.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Configured tool invocation projection via execution-bound wiring resolver",
    ),
    _row(
        "intergrax/tools/marketplace_qualified_capability_execution_handler.py",
        ConfiguredExecutionPathClass.A_CANONICAL,
        "Production marketplace configured handler (execution-bound ingress)",
    ),
    _row(
        "intergrax/runtime/codecraft/qualified_capability_execution_handler.py",
        ConfiguredExecutionPathClass.B_SANCTIONED_NON_CONFIGURED,
        "CodeCraft qualified handler; references adoption type but not CONFIGURED_ADOPTED relational production path",
    ),
    _row(
        "intergrax/tools/marketplace_qualified_capability_execution_composition.py",
        ConfiguredExecutionPathClass.B_SANCTIONED_NON_CONFIGURED,
        "Non-configured marketplace handler composition helper (ordinary qualified path)",
    ),
)
