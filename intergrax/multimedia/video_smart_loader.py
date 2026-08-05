# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
from pathlib import Path

from intergrax.knowledge.contracts import KnowledgeDocument

class VideoSmartLoader:
    """
    Loads video files and converts them into a list of LangChain Documents.

    For each subtitle (VTT) segment:
      - content: transcript text
      - metadata: extracted frame path, mid_time_ms, video_segment_id, etc.

    If a .vtt transcript file is missing, it can automatically generate one
    using Whisper via transcribe_to_vtt().

    It also extracts key video frames using extract_and_frames_and_metadata().
    """

    def __init__(
        self,
        path: str,
        *,
        out_dir: str | None = None,
        frames_subdir: str = "frames",
        meta_subdir: str = "video_meta",
        transcribe_if_missing: bool = True,
        whisper_model_size: str = "base",
        whisper_language: str | None = None,
        frame_target_height: int = 350,
        tenant_id: str = "default",
        namespace: str | None = None,
        workspace_id: str | None = None,
    ):
        """
        Args:
            path: Path to the video file.
            out_dir: Optional directory to save extracted data.
            frames_subdir: Subfolder for extracted frames.
            meta_subdir: Subfolder for metadata.
            transcribe_if_missing: Whether to generate .vtt if missing.
            whisper_model_size: Whisper model name ('tiny', 'base', etc.).
            whisper_language: Optional language code (e.g. 'en', 'pl').
            frame_target_height: Frame resize height while keeping aspect ratio.
        """
        self.path = str(path)
        self._p = Path(path).resolve()

        # Determine output directories
        self.out_root = Path(out_dir) if out_dir else self._p.parent
        self.frames_dir = self.out_root / frames_subdir
        self.meta_dir = self.out_root / meta_subdir

        self.transcribe_if_missing = bool(transcribe_if_missing)
        self.whisper_model_size = whisper_model_size
        self.whisper_language = whisper_language
        self.frame_target_height = int(frame_target_height)
        self.tenant_id = tenant_id
        self.namespace = namespace
        self.workspace_id = workspace_id

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_vtt(self) -> str:
        """
        Checks if a .vtt transcript exists next to the video file.
        If not, optionally generates it using Whisper.
        Returns the absolute path to the .vtt file.
        """
        vtt_path = self._p.with_suffix(".vtt")
        if vtt_path.exists():
            return str(vtt_path)

        if not self.transcribe_if_missing:
            raise FileNotFoundError(f"Missing transcript: {vtt_path}")

        # Generate transcript using your component (Whisper)
        from intergrax.multimedia.video_loader import transcribe_to_vtt

        vtt_path = transcribe_to_vtt(
            input_media_path=str(self._p),
            output_vtt_path=str(vtt_path),
            model_size=self.whisper_model_size,
            language=self.whisper_language,
        )
        return str(vtt_path)

    def load(self) -> list[KnowledgeDocument]:
        """
        Extracts transcript and representative frames per subtitle segment.
        Returns native knowledge documents.
        """
        # 1. Ensure transcript exists
        vtt_path = self._ensure_vtt()

        # 2. Extract frames and metadata using your helper
        from intergrax.multimedia.video_loader import extract_frames_and_metadata

        metas = extract_frames_and_metadata(
            path_to_video=str(self._p),
            path_to_transcript=str(vtt_path),
            path_to_save_extracted_frames=str(self.frames_dir),
            path_to_save_metadatas=str(self.meta_dir),
        )

        source_id = str(self._p)
        root_document_id = "video:" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()
        docs: list[KnowledgeDocument] = []
        for index, m in enumerate(metas):
            transcript = (m.get("transcript") or "").strip()
            if not transcript:
                continue
            segment_id = str(m.get("video_segment_id") or f"segment-{index}")
            document_id = "video-segment:" + hashlib.sha256(
                f"{source_id}:{segment_id}".encode("utf-8")
            ).hexdigest()

            metadata = {
                "doc_type": "video",
                "video_path": m.get("video_path") or str(self._p),
                "video_segment_id": m.get("video_segment_id"),
                "mid_time_ms": m.get("mid_time_ms"),
                "extracted_frame_path": m.get("extracted_frame_path"),
                "transcript_source": "vtt",
            }

            # Include optional timing/frame info if available
            for k in ("start_ms", "end_ms", "start", "end", "frame_index","duration_ms"):
                if k in m and m[k] is not None:
                    metadata[k] = m[k]

            docs.append(
                KnowledgeDocument.model_validate(
                    {
                        "schema_version": 1,
                        "identity": {
                            "document_id": document_id,
                            "root_document_id": root_document_id,
                            "parent_document_id": root_document_id,
                        },
                        "scope": {
                            "tenant_id": self.tenant_id,
                            "namespace": self.namespace,
                            "workspace_id": self.workspace_id,
                        },
                        "content": transcript,
                        "metadata": metadata,
                        "provenance": {
                            "source_kind": "video_segment",
                            "source_id": segment_id,
                            "source_parent_id": source_id,
                        },
                    }
                )
            )

        return docs