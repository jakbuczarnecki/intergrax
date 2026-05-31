# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Only this module may import ``yt_dlp``."""

from __future__ import annotations

from pathlib import Path

from intergrax.integrations.providers.document_parser.yt_dlp.config import YtDlpIntegrationConfig


def download_youtube_audio(config: YtDlpIntegrationConfig, youtube_url: str) -> Path:
    from yt_dlp import YoutubeDL

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / "audio_%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": config.audio_format,
                "preferredquality": "192",
            }
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
    video_id = info.get("id")
    filepath = out_dir / f"audio_{video_id}.{config.audio_format}"
    if not filepath.exists():
        raise FileNotFoundError("Cannot find downloaded audio file.")
    return filepath


def download_youtube_video(config: YtDlpIntegrationConfig, youtube_url: str) -> Path:
    from yt_dlp import YoutubeDL

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / "vid_%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
    video_id = info.get("id")
    filepath = out_dir / f"vid_{video_id}.mp4"
    if not filepath.exists():
        ext = info.get("ext", "mp4")
        alt = out_dir / f"vid_{video_id}.{ext}"
        if alt.exists():
            return alt
        raise FileNotFoundError("Cannot find downloaded video file.")
    return filepath
