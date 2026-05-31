# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Backward-compatible shims — vendor imports live in ``integrations``."""

from __future__ import annotations

from pathlib import Path

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.integrations.providers.document_parser.whisper.config import WhisperIntegrationConfig
from intergrax.integrations.providers.document_parser.whisper.opens import transcribe_audio_file
from intergrax.integrations.providers.document_parser.yt_dlp.config import YtDlpIntegrationConfig
from intergrax.integrations.providers.document_parser.yt_dlp.opens import download_youtube_audio


def yt_download_audio(youtube_url: str, out_dir: str | Path, audio_format: str = "mp3") -> Path:
    config = YtDlpIntegrationConfig(out_dir=Path(out_dir), audio_format=audio_format)
    return download_youtube_audio(config, youtube_url)


def translate_audio(
    audio_path: str,
    model: str = "medium",
    language: str = GLOBAL_SETTINGS.default_language,
):
    config = WhisperIntegrationConfig(model=model, language=language, translate=True)
    return transcribe_audio_file(config, audio_path)
