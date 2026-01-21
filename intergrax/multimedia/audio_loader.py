# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from yt_dlp import YoutubeDL

from intergrax.globals.settings import GLOBAL_SETTINGS

def yt_download_audio(youtube_url: str, out_dir: str | Path, audio_format: str = "mp3") -> Path:
    out_dir = Path(out_dir)
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
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    video_id = info.get("id")
    filepath = out_dir / f"audio_{video_id}.{audio_format}"

    if not filepath.exists():
        raise FileNotFoundError("Cannot find downloaded audio file.")

    return filepath


def translate_audio(audio_path:str, model:str="medium", language:str=GLOBAL_SETTINGS.default_language):
    import whisper
    model = whisper.load_model("medium")
    options = dict(task="translate", best_of=1, language=language)
    results = model.transcribe(str(audio_path), **options)
    return results