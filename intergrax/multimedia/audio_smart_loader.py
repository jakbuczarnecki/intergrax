# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.multimedia.audio_loader import translate_audio, yt_download_audio

class AudioSmartLoader:
    """
    Smart loader for audio files and YouTube audio sources.

    Capabilities:
    - Accepts either a local audio file or a YouTube URL.
    - Downloads the audio (if YouTube URL) using yt_dlp.
    - Transcribes or translates audio using Whisper (translate_audio).
    - Returns a list of LangChain Document objects with metadata per segment.
    """

    def __init__(
        self,
        path: str,
        *,
        out_dir: str | Path | None = None,
        audio_format: str = "mp3",
        whisper_model: str = "medium",
        whisper_language: str = GLOBAL_SETTINGS.default_language,
        translate: bool = True,
    ):
        """
        Args:
            path: Local audio path or YouTube URL.
            out_dir: Directory for downloaded audio files (if YouTube).
            audio_format: Desired output format (mp3, wav, flac, etc.).
            whisper_model: Whisper model name (tiny, base, small, medium, large).
            whisper_language: Language code for transcription/translation.
            translate: If True, translates speech to English (Whisper task="translate").
        """
        self.path = path
        self.out_dir = Path(out_dir or "./audio_downloads")
        self.audio_format = audio_format
        self.whisper_model = whisper_model
        self.whisper_language = whisper_language
        self.translate = translate

    def load(self) -> List[Document]:
        """
        Executes the end-to-end audio pipeline:
        - If path is a YouTube URL → download audio.
        - Run Whisper transcription/translation.
        - Return list of LangChain Documents with metadata.
        """
        audio_path = self._ensure_audio_file()
        transcription = self._transcribe_audio(audio_path)

        # Whisper output example:
        # {
        #   "text": "full transcription",
        #   "segments": [
        #       {"id":0,"start":0.0,"end":3.1,"text":"Hello world"},
        #       ...
        #   ]
        # }

        results = transcription.get("segments", [])
        if not results:
            # fallback: single document with whole text
            return [
                Document(
                    page_content=transcription.get("text", "").strip(),
                    metadata={
                        "source_path": str(audio_path),
                        "source_type": "audio",
                        "language": self.whisper_language,
                        "whisper_model": self.whisper_model,
                    },
                )
            ]

        # Build documents for each segment
        docs: List[Document] = []
        for seg in results:
            seg_text = seg.get("text", "").strip()
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            duration = seg_end - seg_start

            metadata = {
                "source_path": str(audio_path),
                "source_type": "audio",
                "segment_id": seg.get("id"),
                "start_s": seg_start,
                "end_s": seg_end,
                "duration_s": duration,
                "whisper_model": self.whisper_model,
                "language": self.whisper_language,
                "translated": self.translate,
            }

            docs.append(Document(page_content=seg_text, metadata=metadata))

        return docs

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _ensure_audio_file(self) -> Path:
        """If given a YouTube URL, downloads the audio file."""
        if self.path.startswith("http://") or self.path.startswith("https://"):
            # YouTube link → download audio
            print(f"[AudioSmartLoader] Downloading audio from YouTube: {self.path}")
            audio_path = yt_download_audio(
                youtube_url=self.path,
                out_dir=self.out_dir,
                audio_format=self.audio_format,
            )
            print(f"[AudioSmartLoader] Audio downloaded: {audio_path}")
            return audio_path

        # Local file
        audio_path = Path(self.path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return audio_path

    def _transcribe_audio(self, audio_path: Path) -> dict:
        """Transcribes or translates the given audio file using Whisper."""
        print(f"[AudioSmartLoader] Transcribing audio: {audio_path}")
        try:
            result = translate_audio(
                str(audio_path),
                model=self.whisper_model,
                language=self.whisper_language,
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio {audio_path}: {e}")