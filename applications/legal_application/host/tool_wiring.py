# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool catalog wiring for legal_application."""

from __future__ import annotations

from intergrax.applications._shared.modality_wiring import wire_modality_extras
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.modality.modality_profile import ModalityProfile
from intergrax.tools.registry.profile import ToolProfile
from legal_application.host.settings import LegalBackendSettings

_LEGAL_MODALITY_TOOL_IDS = (
    "vision.detect",
    "vision.segment",
    "vision.ocr_regions",
    "speech.synthesize",
    "speech.transcribe",
    "ml.predict",
    "ml.explain",
)


def wire_legal_tools(
    *,
    settings: LegalBackendSettings,
    integration_profile: IntegrationProfile | None = None,
    modality_profile: ModalityProfile | None = None,
) -> ApplicationToolWiring:
    """
    Enable catalog tools declared in :class:`LegalBackendSettings`.

    RAG / websearch require matching ``LEGAL_ENABLE_*`` flags **and** runtime
    managers wired into ``ToolWiringContext`` (vectorstore, embeddings, websearch executor).

    Set ``LEGAL_ENABLE_MODALITY_TOOLS=true`` to enable Plane C modality tools with
    ``VisionProfile`` / ``SpeechProfile`` extras (same pattern as lab harness).
    """
    enabled = list(settings.enabled_tool_ids)
    if settings.enable_modality_tools:
        for tool_id in _LEGAL_MODALITY_TOOL_IDS:
            if tool_id not in enabled:
                enabled.append(tool_id)
    profile = ToolProfile(enabled=enabled) if enabled else ToolProfile()
    wiring = build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
    )
    if settings.enable_modality_tools:
        wire_modality_extras(wiring.wiring_context, modality_profile=modality_profile)
    return wiring
