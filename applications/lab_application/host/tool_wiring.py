# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for lab_application (Phase O.8)."""

from __future__ import annotations

from intergrax.applications._shared.integration_tool_profile import extend_tool_profile_for_integration
from intergrax.applications._shared.integration_tool_wiring import wire_integration_tool_context
from intergrax.applications._shared.modality_wiring import wire_modality_extras
from intergrax.applications._shared.sandbox_host_wiring import resolve_hosted_sandbox_session
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.model_inference.registry import VisionProfile
from intergrax.runtime.modality.modality_profile import ModalityProfile, lab_default_modality_profile
from intergrax.speech_adapters.registry.profile import SpeechProfile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext

_HARNESS_MODALITY_TOOLS = (
    "speech.synthesize",
    "speech.transcribe",
    "vision.detect",
    "vision.segment",
    "vision.ocr_regions",
    "ml.predict",
    "ml.explain",
    "ml.batch_predict",
)


def wire_lab_tools(
    *,
    integration_profile: IntegrationProfile | None = None,
    harness: bool = False,
    sandbox_session: object | None = None,
    modality_profile: ModalityProfile | None = None,
    vision_profile: VisionProfile | None = None,
    speech_profile: SpeechProfile | None = None,
) -> ApplicationToolWiring:
    """
    Laboratory tool wiring — context retrieval tools enabled by default.

    ``sandbox.exec`` is enabled when a sandbox session or ``sandbox_host`` integration
    is wired (U-Sec.3 / M.6 P6).

    Pass ``integration_profile`` from ``wire_lab_integrations()`` when issue/wiki
    tools should resolve integration contracts automatically.
    """
    enabled = ["rag.retrieve", "websearch.query"]
    if sandbox_session is not None:
        enabled.append("sandbox.exec")
    if harness:
        enabled.extend(
            [
                "errors.capture",
                "observability.query_traces",
                "pagerduty.trigger_incident",
                "gitlab.create_issue",
                "braintrust.log_eval",
                *_HARNESS_MODALITY_TOOLS,
            ]
        )
    resolved_modality = modality_profile
    if harness and resolved_modality is None:
        resolved_modality = lab_default_modality_profile()
    profile = ToolProfile(enabled=enabled)
    profile = extend_tool_profile_for_integration(profile, integration_profile)

    wiring_context = ToolWiringContext()
    if integration_profile is not None:
        wiring_context = ToolWiringContext.from_integration_profile(integration_profile)
        wiring_context = wire_integration_tool_context(wiring_context, integration_profile)

    hosted_session = None
    if sandbox_session is None and integration_profile is not None:
        hosted_session = resolve_hosted_sandbox_session(
            integration_profile,
            tenant_id="lab",
            task_id="mcp-tools",
        )

    wiring = build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
        wiring_context=wiring_context,
        sandbox_session=sandbox_session or hosted_session,
    )
    if harness:
        wire_modality_extras(
            wiring.wiring_context,
            modality_profile=resolved_modality,
            vision_profile=vision_profile,
            speech_profile=speech_profile,
        )
    return wiring
