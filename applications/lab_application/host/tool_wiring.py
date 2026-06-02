# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for lab_application (Phase O.8)."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.runtime.modality.modality_profile import (
    MODALITY_PROFILE_EXTRA_KEY,
    ModalityProfile,
    lab_default_modality_profile,
)
from intergrax.tools.providers.speech.backends import (
    MODEL_INFERENCE_REGISTRY_EXTRA_KEY,
    SPEECH_BACKEND_EXTRA_KEY,
    build_speech_backend,
)
from intergrax.tools.registry.profile import ToolProfile


def wire_lab_tools(
    *,
    integration_profile: IntegrationProfile | None = None,
    harness: bool = False,
    sandbox_session: object | None = None,
    modality_profile: ModalityProfile | None = None,
) -> ApplicationToolWiring:
    """
    Laboratory tool wiring — context retrieval tools enabled by default.

    ``sandbox.exec`` is enabled only when a sandbox session is wired (U-Sec.3).

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
                "speech.synthesize",
                "speech.transcribe",
                "vision.detect",
                "ml.predict",
            ]
        )
    resolved_modality = modality_profile
    if harness and resolved_modality is None:
        resolved_modality = lab_default_modality_profile()
    profile = ToolProfile(enabled=enabled)
    wiring = build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
        sandbox_session=sandbox_session,
    )
    if resolved_modality is not None:
        wiring.wiring_context.extras[MODALITY_PROFILE_EXTRA_KEY] = resolved_modality
    if harness:
        wiring.wiring_context.extras[SPEECH_BACKEND_EXTRA_KEY] = build_speech_backend()
        wiring.wiring_context.extras[MODEL_INFERENCE_REGISTRY_EXTRA_KEY] = (
            build_harness_model_inference_registry()
        )
    return wiring
