# © Artur Czarnecki. All rights reserved.

from intergrax.tools.providers.speech.bundle import SPEECH_BUNDLE_ID, register_speech_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_speech_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=SPEECH_BUNDLE_ID,
            tool_ids=("speech.synthesize", "speech.transcribe"),
            register=register_speech_tools,
            status=ToolBundleStatus.STABLE,
            description="Speech synthesis and transcription tools (Plane B/C bridge).",
        ),
        override=override,
    )
