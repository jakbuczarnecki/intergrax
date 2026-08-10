# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Only this module may import ``whisper``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.whisper.config import (
    WhisperIntegrationConfig,
)
from intergrax.integrations.providers.document_parser.yt_dlp.config import (
    YtDlpIntegrationConfig,
)
from intergrax.integrations.providers.document_parser.yt_dlp.opens import (
    download_youtube_audio,
)


def _import_whisper() -> Any:
    try:
        import whisper
    except ModuleNotFoundError as exc:
        if exc.name in {"whisper", "torch"}:
            raise IntegrationDependencyError(
                "Provider 'whisper' requires optional dependency 'openai-whisper'. "
                "Install Intergrax-ai[media-whisper].",
                integration_name="whisper",
            ) from exc
        raise
    return whisper


def transcribe_audio_file(config: WhisperIntegrationConfig, audio_path: str | Path) -> dict[str, Any]:
    whisper = _import_whisper()
    model = whisper.load_model(config.model)
    task = "translate" if config.translate else "transcribe"
    options: dict[str, Any] = {
        "task": task,
        "best_of": 1,
        "language": config.language,
    }
    return model.transcribe(str(audio_path), **options)


def _resolve_audio_path(config: WhisperIntegrationConfig, source: str) -> Path:
    if source.startswith(("http://", "https://")):
        yt_config = YtDlpIntegrationConfig(
            out_dir=config.out_dir,
            audio_format=config.audio_format,
        )
        return download_youtube_audio(yt_config, source)
    audio_path = Path(source)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path


def whisper_is_available() -> bool:
    try:
        _import_whisper()
        return True
    except Exception:  # noqa: BLE001 - availability probes must fail closed
        return False


def parse_whisper_audio(config: WhisperIntegrationConfig, source: str) -> list[ParsedDocumentFragment]:
    audio_path = _resolve_audio_path(config, source)
    transcription = transcribe_audio_file(config, audio_path)
    segments = transcription.get("segments", [])
    if not segments:
        text = (transcription.get("text") or "").strip()
        if not text:
            return []
        return [
            ParsedDocumentFragment(
                text=text,
                metadata={
                    "parser_backend": "whisper",
                    "source_path": str(audio_path),
                    "whisper_model": config.model,
                    "language": config.language,
                },
            )
        ]
    fragments: list[ParsedDocumentFragment] = []
    for seg in segments:
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue
        fragments.append(
            ParsedDocumentFragment(
                text=seg_text,
                metadata={
                    "parser_backend": "whisper",
                    "source_path": str(audio_path),
                    "segment_id": seg.get("id"),
                    "start_s": float(seg.get("start", 0)),
                    "end_s": float(seg.get("end", 0)),
                    "whisper_model": config.model,
                    "language": config.language,
                    "translated": config.translate,
                },
            )
        )
    return fragments


def transcribe_media_to_vtt(
    config: WhisperIntegrationConfig,
    input_media_path: str | Path,
    output_vtt_path: Path | None = None,
) -> Path:
    import webvtt
    from tqdm.auto import tqdm

    whisper = _import_whisper()
    input_media_path = Path(input_media_path)
    target = output_vtt_path or input_media_path.with_suffix(".vtt")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _sec_to_vtt_ts(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"

    if not target.exists():
        model = whisper.load_model(config.model)
        result = model.transcribe(str(input_media_path), language=config.language)
        vtt = webvtt.WebVTT()
        for segment in tqdm(result.get("segments", []), desc="Transcribing", unit="seg"):
            start = _sec_to_vtt_ts(segment["start"])
            end = _sec_to_vtt_ts(segment["end"])
            text = " ".join(segment.get("text", "").split())
            vtt.captions.append(webvtt.Caption(start, end, text))
        vtt.save(str(target))
    return target
