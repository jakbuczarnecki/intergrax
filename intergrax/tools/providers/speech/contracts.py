# © Artur Czarnecki. All rights reserved.

"""Re-export speech I/O contracts from Tier-0 ``speech_adapters``."""

from intergrax.speech_adapters.contracts.io import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)

__all__ = [
    "SpeechSynthesizeInput",
    "SpeechSynthesizeOutput",
    "SpeechTranscribeInput",
    "SpeechTranscribeOutput",
]
